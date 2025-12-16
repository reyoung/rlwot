import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
import typing
from pathlib import Path
from typing import List
import random
import contextvars
from .cluster import *

import datasets
import torch
import transformers
import yaml
import json
import tqdm
from .dapo_utils import verify as verify_dapo
logger = logging.getLogger(__name__)


WorkerSeed = contextvars.ContextVar("worker_seed")


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

        lora_a = torch.zeros((config.lora_r, cols), dtype=torch.bfloat16)
        lora_b = torch.zeros((rows, config.lora_r), dtype=torch.bfloat16)

        res[f"{prefix}.lora_A.weight"] = lora_a
        res[f"{prefix}.lora_B.weight"] = lora_b

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
    seed: int | None = WorkerSeed.get(None)
    if seed is not None:
        with open(f"eval_logs/seed_{seed}_eval.jsonl", "a") as outf:
            worker_id = worker.id()
            json.dump({
                "lora_id": worker_id,
                "request": req.messages,
                "generation_args": req.args.model_dump() if req.args is not None else None,
                "response": response,
                "ground_truth": ground_truth,
                "ok": ok,
            }, outf)
            outf.write("\n")


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

def _generate_lora_noise(seed: int, m: int, n: int, k: int, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        with torch.random.fork_rng(devices=["cpu"]):
            torch.manual_seed(seed)
            A = torch.randn((k, n)) * sigma
            B = torch.randn((m, k))
        return A, B

def _extract_weight_names(base_model: dict[str, torch.Tensor]) -> list[str]:
    weight_names = [k.removesuffix(".lora_A.weight") for k in base_model.keys() if k.endswith(".lora_A.weight")]
    weight_names.sort()
    return weight_names


def _generate_noise(seed: int, base_model: dict[str, torch.Tensor], sigma: float):
    weight_names = _extract_weight_names(base_model)
    # logger.info(f"Generating noise for {weight_names} weights")
    noise = {}
    for offset, weight_name in enumerate(weight_names):
        a = base_model[f"{weight_name}.lora_A.weight"]
        b = base_model[f"{weight_name}.lora_B.weight"]
        k, n= a.shape
        m, _ = b.shape
        noise_a, noise_b = _generate_lora_noise(seed + offset, m, n, k, sigma)
        noise[f"{weight_name}.lora_A.weight"] = noise_a
        noise[f"{weight_name}.lora_B.weight"] = noise_b
    
    return noise

async def _calc_worker_gradient(semaphore: asyncio.Semaphore, 
                                seed: int, 
                                base_model: dict[str, torch.Tensor], 
                                cfg: TrainConfig, 
                                cluster: Cluster,
                                dataset: typing.Callable[[],typing.Generator[tuple[ChatCompletionRequest, str], None, None]],
                                pbar: tqdm.tqdm,
                                worker_id: int) -> WorkerGradient:
    WorkerSeed.set(seed)
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
            # symmetric noise
            new_model = {}
            for k, v in base_model.items():
                if k.endswith(".lora_A.weight"):
                    new_model[k] = v - noise[k]
                else:
                    new_model[k] = v + noise[k]
        
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

@torch.no_grad()
def extract_lora_weights(mat: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract LoRA weights from a matrix using SVD decomposition.
    
    Args:
        mat: Input matrix of shape (m, n)
        rank: Target rank for the decomposition
        
    Returns:
        Tuple of (lora_A, lora_B) where lora_A has shape (m, rank) and lora_B has shape (rank, n)
        such that lora_A @ lora_B approximates mat
    """
    # Perform SVD decomposition: mat = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
    
    # Take only the top 'rank' singular values
    U_r = U[:, :rank]  # (m, rank)
    S_r = S[:rank]     # (rank,)
    Vt_r = Vt[:rank, :]  # (rank, n)
    
    # Split the singular values between A and B
    # A = U_r @ sqrt(S_r), B = sqrt(S_r) @ Vt_r
    sqrt_S_r = torch.sqrt(S_r)
    
    lora_A = sqrt_S_r.unsqueeze(1) * Vt_r  # (rank, n)
    lora_B = U_r * sqrt_S_r.unsqueeze(0)  # (m, rank)
    
    return lora_A, lora_B



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
                logger.info("acquiring semaphore for worker %d/%d, sema counter %d", worker_id, len(worker_seeds) ,cfg.max_concurrent_workers)
                await semaphore.acquire()
                logger.info("acquired semaphore for worker %d", worker_id)
                
                worker_grads_tasks.append(
                    tg.create_task(_calc_worker_gradient(
                        semaphore,
                        worker_seed, base_model, cfg, cluster,
                        lambda: dataset_to_generator(train_dataset, worker_id, cfg.n_workers),
                        pbar,
                        worker_id
                    ))
                )
            worker_grads: list[WorkerGradient] = await asyncio.gather(*worker_grads_tasks)
        
        positive_scores = sum(wg.positive_score for wg in worker_grads) / len(worker_grads)
        negative_scores = sum(wg.negative_score for wg in worker_grads) / len(worker_grads)
        logger.info(f"Epoch {epoch_id} positive score {positive_scores} negative score {negative_scores}")

        weight_names = _extract_weight_names(base_model)
        with torch.no_grad():
            for offset, weight_name in enumerate(weight_names):
                origin_a = base_model[f"{weight_name}.lora_A.weight"]
                origin_b = base_model[f"{weight_name}.lora_B.weight"]
                k, n = origin_a.shape
                m, k = origin_b.shape

                logger.info("updating weight %s, m,n,k = %d,%d,%d", weight_name, m, n, k)
                update = torch.zeros((m, n))
                for wg in worker_grads:
                    noise_a, noise_b = _generate_lora_noise(wg.seed + offset, m, n, k, cfg.sigma)

                    positive_noise = noise_b @ noise_a
                    negative_noise = -noise_b @ noise_a

                    positive_score = wg.positive_score
                    negative_score = wg.negative_score

                    update += (positive_noise - negative_noise) * (positive_score - negative_score) / 2.0
                
                lora_a_update, lora_b_update = extract_lora_weights(update, rank=k)
                base_model[f"{weight_name}.lora_A.weight"] += lora_a_update
                base_model[f"{weight_name}.lora_B.weight"] += lora_b_update

                abs_error = torch.mean(torch.abs(update -  lora_b_update@lora_a_update))
                logger.info(f"Weight {weight_name} abs error {abs_error}")



        with torch.no_grad():
            torch.save(base_model, f"./model_{epoch_id}.pt")
            

async def amain(config: Config):
    # Configure logging to both file and stderr
    log_level = logging.DEBUG if config.debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(config.save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    log_filename = os.path.join(log_dir, f"rlwot_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Create stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(log_level)
    stderr_handler.setFormatter(logging.Formatter(log_format))
    
    # Add both handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stderr_handler)
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info(f"Logging to file: {log_filename}")
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
        
def handle_exception(loop, context):
    exc = context.get("exception")
    if exc:
        print("Unhandled exception:", exc)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    else:
        print("Unhandled error:", context.get("message"))
        
def main() -> None:
    """Main entry point for script."""
    config = parse_args()
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)
    loop.run_until_complete(amain(config))
    


if __name__ == "__main__":
    main()
