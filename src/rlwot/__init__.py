import argparse
import asyncio
import collections
import contextlib
import functools
import logging
import os
import subprocess
import sys
import time
import typing
from uuid_extensions import uuid7
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, List, Literal
import random
from dataclasses import dataclass

import datasets
import httpx
import pydantic
import torch
import transformers
import yaml
import json
import tqdm
from .dapo_utils import verify as verify_dapo
logger = logging.getLogger(__name__)

ChatCompletionMessages = list[dict[str, str]]
class GenerateArgs(pydantic.BaseModel):
    recursive: bool = pydantic.Field(default=True)
    max_tokens: int = pydantic.Field(strict=True, gt=0, default=256)
    seed: int | None = None

@dataclass
class ChatCompletionRequest:
    messages: ChatCompletionMessages
    args: GenerateArgs | None  = None


    def transform(self, fn: typing.Callable[[typing.Self], typing.Self]) -> typing.Self:
        return fn(self)


class StandaloneVLLMClusterConfig(pydantic.BaseModel):
    port: Annotated[int, pydantic.Field(strict=True, gt=0, lt=65536)] = pydantic.Field(
        default=58000, description="Port number for the standalone VLLM cluster"
    )
    type: Literal["standalone_vllm"] = pydantic.Field(
        default="standalone_vllm", description="Type of the cluster"
    )
    start_timeout_in_sec: Annotated[int, pydantic.Field(strict=True, gt=0)] = (
        pydantic.Field(
            default=600, description="Timeout in seconds for starting the VLLM server"
        )
    )
    lora_dir: str = pydantic.Field(
        default="lora", description="Directory for storing LoRA models"
    )
    max_model_len: int = 1024 * 6

    def as_args(self) -> typing.Generator[str, None, None]:
        yield "--port"
        yield str(self.port)
        yield "--max-model-len"
        yield str(self.max_model_len)

class VLLMClusterConfig(pydantic.BaseModel):
    addresses: list[str] = pydantic.Field([], description="Addresses of VLLM servers")
    type: Literal["vllm"] = pydantic.Field(default="vllm", description="Type of the cluster")
    lora_dir: str = pydantic.Field(
        default="lora", description="Directory for storing LoRA models"
    )

ClusterConfig = Annotated[
    typing.Union[StandaloneVLLMClusterConfig, VLLMClusterConfig],
    pydantic.Field(discriminator="type"),
]


class TrainConfig(pydantic.BaseModel):
    seed: int = 42
    n_epochs: int = 100
    n_rollouts_per_sample: int = 5
    n_workers: int = 10
    concurrency: int = 10
    max_concurrent_workers: int = 2
    n_samples: int | None = None
    sigma: float = 0.2

    @property
    def eval_concurrency(self):
        return self.concurrency * self.n_workers


class Config(pydantic.BaseModel):
    """Configuration for LoRA model generation."""

    base_model: str = pydantic.Field(
        default="Qwen/Qwen3-4B", description="Base model name"
    )

    target_modules: List[str] = pydantic.Field(
        default=[
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "v_proj",
            "k_proj",
        ],
        description="Target modules to apply LoRA",
    )

    lora_r: Annotated[int, pydantic.Field(strict=True, gt=1)] = pydantic.Field(
        default=8, description="Rank of LoRA"
    )

    save_dir: str = pydantic.Field(
        default="ckpt", description="Directory to save the LoRA model"
    )

    cluster_config: ClusterConfig = pydantic.Field(
        default=StandaloneVLLMClusterConfig(), description="Cluster configuration"
    )

    debug: bool = pydantic.Field(default=False, description="Debug mode")

    train: TrainConfig = pydantic.Field(
        default=TrainConfig(), description="Training configuration"
    )

    class Config:
        extra = "forbid"  # Forbid extra fields in the config

    def model_post_init(self, __context) -> None:
        """Create the save directory if it doesn't exist."""
        os.makedirs(self.save_dir, exist_ok=True)


class Cluster(typing.Protocol):
    async def start(self):
        raise NotImplementedError("start is not implemented")

    def close(self) -> None:
        raise NotImplementedError("close is not implemented")

    async def new_worker(self, states: dict[str, torch.Tensor]) -> "Worker":
        raise NotImplementedError("new_worker is not implemented")

class Worker(typing.Protocol):
    async def close(self) -> None: 
        raise NotImplementedError("close is not implemented")


    async def chat_completion(self, messages: ChatCompletionMessages, args: GenerateArgs | None = None) -> str:
        raise NotImplementedError("chat_completion is not implemented")


@contextlib.asynccontextmanager
async def new_worker(cluster: Cluster, states: dict[str, torch.Tensor]):
    worker = await cluster.new_worker(states)
    try:
        yield worker
    finally:
        try:
            await worker.close()
        except Exception as e:
            logger.error(f"Error closing worker: {e}")
            raise


def _save_lora_model(
    model_path: str, r: int, target_modules: list[str], states: dict[str, torch.Tensor]
): 
    configs = {
        "r": r,
        "target_modules": target_modules,
        "lora_alpha": 1, # alpha is not important for us
    }
    with open(os.path.join(model_path, "adapter_config.json"), "w") as f:
        json.dump(configs, f, indent=4)

    with open(os.path.join(model_path, "adapter_model.pt"), "wb") as f:
        torch.save(states, f)


class VLLMWorker(Worker):
    def __init__(self, worker_id: str,  client: httpx.AsyncClient, addr: str, chat_template: "_ChatTemplate") -> None:
        self._worker_id = worker_id
        self._http_client = client
        self._addr = addr
        self._chat_template = chat_template

    async def close(self) -> None:
        resp = await self._http_client.post(
            f"{self._addr}/v1/unload_lora_adapter",
            json={
                "lora_name": self._worker_id,
            }
        )
        resp.raise_for_status()


    async def chat_completion(self, messages: List[dict[str, str]], args: GenerateArgs | None = None) -> str:
        if args is None:
            args = GenerateArgs()
        prompt: str = await self._chat_template.apply_chat_template(messages, add_generation_token=True)
        response: str = ""

        while True:
            request_json = {
                "model": self._worker_id,
                "prompt": prompt + response,
                "max_tokens": args.max_tokens,
            }
            if args.seed is not None:
                request_json["seed"] = args.seed

            resp = await self._http_client.post(
                f"{self._addr}/v1/completions",
                json=request_json
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                if resp.status_code / 100 == 4:
                    # 4xx is client error
                    logger.info("ignore request vllm error %d: %s", resp.status_code,  resp.text)
                    return response
                raise 
            response_json = resp.json()

            c0 = response_json["choices"][0]
            response += c0["text"]

            if c0["finish_reason"] != "length":
                return response


class VLLMCluster(Cluster):
    def __init__(self, config: VLLMClusterConfig, base_cfg: Config):
        self._config = config
        self._base_config = base_cfg
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(read=1800, connect=5, write=5, pool=5),
                                              limits=httpx.Limits(max_connections=sys.maxsize, max_keepalive_connections=sys.maxsize))
        self._pool = ThreadPoolExecutor(max_workers=8)
        self._chat_template = _ChatTemplate(self._base_config.base_model, self._pool)
        self._next_addr_idx = 0
        self._addr_counter = collections.defaultdict(int)


    async def start(self):
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(self._health_check(addr)) for addr in self._config.addresses]
            await asyncio.gather(*tasks)
    
    async def _health_check(self, addr: str):
        begin = time.time()
        while time.time() - begin < 60:
            try:
                resp = await self._http_client.get(f"{addr}/health")
            except httpx.RequestError:
                time.sleep(0.5)
                continue
            resp.raise_for_status()
            break

        test_prompt = ["Hello world!"]
        response = await self._http_client.post(
            f"{addr}/v1/completions",
            json={
                    "model": "base",
                    "prompt": test_prompt,
                    "max_tokens": 7,
                },
        )
        response.raise_for_status()
        logger.debug("recv response %s", response.json())

    def close(self):
        self._pool.shutdown(wait=True)

    async def new_worker(self, states: dict[str, torch.Tensor]) -> "Worker":
        worker_id = str(uuid7())
        worker_dir = os.path.join(self._config.lora_dir, worker_id)
        os.makedirs(worker_dir, exist_ok=True)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._pool, _save_lora_model, worker_dir, self._base_config.lora_r, 
                             self._base_config.target_modules, states)
        
        addr = self._config.addresses[self._next_addr_idx]
        self._next_addr_idx = (self._next_addr_idx + 1) % len(self._config.addresses)
        self._addr_counter[addr] += 1
        counter = self._addr_counter[addr]
        logger.info(f"Assigning new worker {worker_id} to address {addr} (total workers on this addr: {counter})")
    
        resp = await self._http_client.post(
            f"{addr}/v1/load_lora_adapter",
            json={
                "lora_name": worker_id,
                "lora_path": worker_dir,
            }
        )
        resp.raise_for_status()
        return VLLMWorker(
            worker_id=worker_id,
            client=self._http_client,
            addr=addr,
            chat_template=self._chat_template,
        )


class _ChatTemplate:
    def __init__(self, model_name: str, pool: ThreadPoolExecutor) -> None:
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._pool = pool

    def _apply_chat_template(self, messages: ChatCompletionMessages, add_generation_token=False) -> str:
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_token=add_generation_token)
    
    async def apply_chat_template(self, messages: ChatCompletionMessages, add_generation_token=False) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool,
            self._apply_chat_template,
            messages,
            add_generation_token,
        )



class StandaloneVLLMCluster(Cluster):
    def __init__(self, config: StandaloneVLLMClusterConfig, base_cfg: Config):
        self._config = config
        self._base_config = base_cfg
        self._vllm_process: subprocess.Popen | None = None
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(read=1800, connect=5, write=5, pool=5))
        self._pool = ThreadPoolExecutor(max_workers=8)
        self._chat_template = _ChatTemplate(self._base_config.base_model, self._pool)
        
    @functools.cached_property
    def addr(self) -> str:
        return f"http://127.0.0.1:{self._config.port}"


    async def start(self):
        args = [
            "vllm",
            "serve",
            self._base_config.base_model,
            "--enable-lora",
            "--served-model-name",
            "base",
            "--max-lora-rank",
            str(self._base_config.lora_r),
            "--max-loras",
            str(self._base_config.train.n_workers),
            "--uvicorn-log-level",
            "error",
            # "--speculative-config",
            # json.dumps({"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4})
        ]
        args.extend(self._config.as_args())
        logger.info(f"Starting VLLM server with args: {args}")
        env = os.environ.copy()
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        self._vllm_process = subprocess.Popen(args, env=env)

        # Waiting for the server is started
        try:
            await self._wait_for_server()
        except:
            self._stop(force=True)
            raise


    async def _wait_for_server(self):
        begin = time.time()
        while time.time() - begin < self._config.start_timeout_in_sec:
            try:
                await self._http_client.get(f"{self.addr}/health")
                break
            except httpx.RequestError:
                time.sleep(0.1)
                assert self._vllm_process is not None
                if self._vllm_process.poll() is not None:
                    raise RuntimeError("VLLM server failed to start")
        else:
            raise TimeoutError(
                f"VLLM server did not start within {self._config.start_timeout_in_sec} seconds"
            )

        # call openai completion API
        test_prompt = ["Hello world!"]
        response = await self._http_client.post(
            f"{self.addr}/v1/completions",
            json={
                "model": "base",
                "prompt": test_prompt,
                "max_tokens": 7,
            },
        )
        response.raise_for_status()
        logger.debug("recv response %s", response.json())

    def _stop(self, force=False):
        if self._vllm_process is not None:
            if not force:
                self._vllm_process.terminate()
            else:
                self._vllm_process.kill()
            self._vllm_process.wait()
            self._vllm_process = None

        self._pool.shutdown(wait=True)

    def close(self):
        self._stop()

    async def new_worker(self, states: dict[str, torch.Tensor]) -> "Worker":
        worker_id = str(uuid7())
        worker_dir = os.path.join(self._config.lora_dir, worker_id)
        os.makedirs(worker_dir, exist_ok=True)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._pool, _save_lora_model, worker_dir, self._base_config.lora_r, 
                             self._base_config.target_modules, states)
        
        resp = await self._http_client.post(
            f"{self.addr}/v1/load_lora_adapter",
            json={
                "lora_name": worker_id,
                "lora_path": worker_dir,
            }
        )
        resp.raise_for_status()
        return VLLMWorker(
            worker_id=worker_id,
            client=self._http_client,
            addr=self.addr,
            chat_template=self._chat_template,
        )



@contextlib.asynccontextmanager
async def connect_cluster(
    config: Config
):
    if isinstance(config.cluster_config, StandaloneVLLMClusterConfig):
        cluster = StandaloneVLLMCluster(config.cluster_config, config)
    elif isinstance(config.cluster_config, VLLMClusterConfig):
        cluster = VLLMCluster(config.cluster_config, config)
    else:
        raise ValueError(f"Unsupported cluster configuration: {config}")

    await cluster.start()
    try:
        yield cluster
    finally:
        cluster.close()


def parse_args() -> Config:
    """Parse command line arguments and load configuration from file."""
    parser = argparse.ArgumentParser(description="Generate LoRA model configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {args.config}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return Config(**config_dict)
    else:
        return Config()


def _name_in_target_modules(name: str, targets: List[str]) -> bool:
    return any(target in name for target in targets)


def generate_default_lora_model(config: Config) -> dict[str, torch.Tensor]:
    """Generate a default LoRA model configuration.

    Args:
        config: Config object containing model parameters

    Returns:
        Dictionary containing LoRA tensors
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model)
    res = {}

    for name, param in model.named_parameters():
        if not _name_in_target_modules(name, config.target_modules):
            continue

        # Only process weight parameters, not biases
        if not name.endswith(".weight"):
            continue

        assert param.dim() == 2, f"{name} is not a linear layer"

        rows, cols = param.shape

        logger.debug(f"generating lora for {name} with shape {rows}x{cols}")
        assert name.endswith(".weight"), "name must end with .weight"

        # base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
        prefix = name[: -len(".weight")]

        lora_a = torch.zeros((rows, config.lora_r), dtype=torch.bfloat16)
        lora_b = torch.zeros((config.lora_r, cols), dtype=torch.bfloat16)

        res[f"base.{prefix}.lora_A.weight"] = lora_a
        res[f"base.{prefix}.lora_B.weight"] = lora_b

    # Save the LoRA tensors to the specified directory
    save_path = os.path.join(config.save_dir, "lora_tensors.pt")
    torch.save(res, save_path)
    logger.info(f"Saved LoRA tensors to {save_path}")

    return res


def dataset_to_generator(
    dataset: datasets.Dataset,
    rank: int,
    world_size: int,
) -> typing.Generator[tuple[ChatCompletionRequest, str], None, None]:
    # Use proper data sharding to ensure equal distribution
    total_samples = len(dataset)
    chunk_size = total_samples // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else total_samples
    
    for sample_id in range(start_idx, end_idx):
        sample = dataset[sample_id]
        yield ChatCompletionRequest(messages=sample["prompt"]), sample["reward_model"]["ground_truth"] # type: ignore


async def eval_sample(
    worker: Worker,
    sample: tuple[ChatCompletionRequest, str],
) -> float:
    req, ground_truth = sample
    response = await worker.chat_completion(req.messages, req.args)
    ok, _ = verify_dapo(response, ground_truth)
    return float(ok)

def _attach_default_gen_args(req: ChatCompletionRequest) -> ChatCompletionRequest:
    if req.args is None:
        req.args = GenerateArgs()
    return req

class _FixGenerateSeed:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
    
    def __call__(self, req: ChatCompletionRequest) -> ChatCompletionRequest:
        next_seed = self._rng.randint(0, 2**32 - 1)
        assert req.args is not None
        req.args.seed = next_seed
        return req


async def _producer(
    data: typing.Generator[tuple[ChatCompletionRequest, str], None, None],
    job_queue: asyncio.Queue[tuple[ChatCompletionRequest, str] | None],
    concurrency: int,
    seed: int | None = None
):
    trms = [_attach_default_gen_args]
    if seed is not None:
        trms.append(_FixGenerateSeed(seed))

    for sample in data:
        req, ground_truth = sample
        for trm in trms:
            req = req.transform(trm)

        await job_queue.put((req, ground_truth))

    for c in range(concurrency):
        await job_queue.put(None)


async def _consumer(
    worker: Worker,
    job_queue: asyncio.Queue[tuple[ChatCompletionRequest, str] | None],
    result_queue: asyncio.Queue[float | None],
):
    while (sample := await job_queue.get()) is not None:
        score = await eval_sample(worker, sample)
        await result_queue.put(score)

    await result_queue.put(None)


async def eval(
    cluster: Cluster,
    model: dict[str, torch.Tensor],
    data: typing.Generator[tuple[ChatCompletionRequest, str], None, None],
    concurrency: int,
    rollout_seed: int | None = None,
    pbar: tqdm.tqdm | None = None,
) -> float:
    async with new_worker(cluster, model) as worker, asyncio.TaskGroup() as tg:
        job_queue = asyncio.Queue(concurrency)
        result_queue = asyncio.Queue(concurrency)

        tg.create_task(_producer(data, job_queue, concurrency, seed=rollout_seed))
        for _ in range(concurrency):
            tg.create_task(_consumer(worker, job_queue, result_queue))

        sum_scores = 0
        n_scores = 0
        while concurrency != 0:
            score = await result_queue.get()
            if score is None:
                concurrency -= 1
            else:
                pbar.update(1) if pbar is not None else None
                sum_scores += score
                n_scores += 1

        return sum_scores / n_scores

class WorkerGradient(typing.NamedTuple):
    seed: int
    positive_score: float
    negative_score: float

def _generate_noise(seed: int, base_model: dict[str, torch.Tensor], sigma: float):
    # rng = random.Random(seed)
    with torch.no_grad():
        with torch.random.fork_rng(devices=["cpu"]):
            torch.manual_seed(seed)
        new_model = {}
        for k, v in base_model.items():
            noise = torch.randn_like(v) * sigma # Small noise for exploration
            new_model[k] = noise
        return new_model


async def _calc_worker_gradient(semaphore: asyncio.Semaphore, 
                                seed: int, 
                                base_model: dict[str, torch.Tensor], 
                                cfg: TrainConfig, 
                                cluster: Cluster,
                                dataset: typing.Callable[[],typing.Generator[tuple[ChatCompletionRequest, str], None, None]],
                                pbar: tqdm.tqdm) -> WorkerGradient:
    try:
        noise = _generate_noise(seed, base_model, sigma=cfg.sigma)

        with torch.no_grad():
            new_model = {k: base_model[k] + noise[k] for k in base_model.keys()}


        positive_score = await eval(cluster=cluster, 
                                model=new_model, 
                                data=dataset(), 
                                concurrency=cfg.concurrency, 
                                rollout_seed=seed,
                                pbar=pbar)
        with torch.no_grad():
            new_model = {k: base_model[k] - noise[k] for k in base_model.keys()}
        
        negative_score = await eval(cluster=cluster, 
                                model=new_model, 
                                data=dataset(), 
                                concurrency=cfg.concurrency, 
                                rollout_seed=seed,
                                pbar=pbar)

        wg = WorkerGradient(seed=seed, positive_score=positive_score, negative_score=negative_score)
        logger.info(f"Worker seed {seed} positive score {positive_score} negative score {negative_score}")
        return wg
    finally:
        semaphore.release()


async def train_loop(
    cluster: Cluster,
    base_model: dict[str, torch.Tensor],
    train_dataset: datasets.Dataset,
    eval_dataset: datasets.Dataset,
    cfg: TrainConfig,
):
    train_dataset = train_dataset.repeat(num_times=cfg.n_rollouts_per_sample)
    rng = random.Random(cfg.seed)
    for epoch_id in range(cfg.n_epochs):
        train_dataset.shuffle(seed=rng.randint(0, 2**32 - 1))
        # train_dataset.
        worker_seeds = [rng.randint(0, 2**32 - 1) for _ in range(cfg.n_workers)]
        pbar = tqdm.tqdm(desc=f"Epoch {epoch_id} Training", total=len(train_dataset) * 2) # * 2 cause by positive and negative
        
        async with asyncio.TaskGroup() as tg:
            worker_grads_tasks = []
            semaphore = asyncio.Semaphore(cfg.max_concurrent_workers)

            for worker_id, worker_seed in enumerate(worker_seeds):
                logger.info("acquiring semaphore for worker %d", worker_id)
                await semaphore.acquire()
                
                worker_grads_tasks.append(
                    tg.create_task(_calc_worker_gradient(
                        semaphore,
                        worker_seed, base_model, cfg, cluster,
                        lambda: dataset_to_generator(train_dataset, worker_id, cfg.n_workers),
                        pbar
                    ))
                )
            worker_grads = await asyncio.gather(*worker_grads_tasks)
        
        positive_scores = sum(wg.positive_score for wg in worker_grads) / len(worker_grads)
        negative_scores = sum(wg.negative_score for wg in worker_grads) / len(worker_grads)
        for wg in worker_grads:
            logger.info(f"Epoch {epoch_id} Worker seed {wg.seed} positive score {wg.positive_score} negative score {wg.negative_score}")

            diff = _generate_noise(wg.seed, base_model, sigma=cfg.sigma)
            score_diff = wg.positive_score - wg.negative_score
            with torch.no_grad():
                update_norms = []
                for k in base_model.keys():
                    update = diff[k] * (score_diff / 2.0)
                    update_norms.append(torch.norm(update, 2).item())
                    base_model[k] += update
                logger.info(f"Epoch {epoch_id} Worker seed {wg.seed} update norm {torch.norm(torch.tensor(update_norms), 2).item()}")

        logger.info(f"Epoch {epoch_id} positive score {positive_scores} negative score {negative_scores}")
        with torch.no_grad():
            torch.save(base_model, f"./model_{epoch_id}.pt")
            

async def amain(config: Config):
    logging.basicConfig(level=logging.DEBUG if config.debug else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    async with connect_cluster(
        config
    ) as cluster:
        base_model = generate_default_lora_model(config)

        dataset: datasets.DatasetDict = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", "default") # type: ignore
        train_dataset:  datasets.Dataset = dataset["train"]
        if config.train.n_samples is not None:
            train_dataset = train_dataset.select(range(config.train.n_samples))
        eval_dataset = train_dataset

        await train_loop(cluster, base_model, train_dataset, eval_dataset, config.train)
        

def main() -> None:
    """Main entry point for script."""
    config = parse_args()
    asyncio.run(amain(config))
    


if __name__ == "__main__":
    main()
