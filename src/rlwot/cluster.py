import asyncio
import collections
from concurrent.futures import ThreadPoolExecutor
import contextlib
import functools
import json
import os
import random
import subprocess
import sys
import time
import typing
from uuid_extensions import uuid7
import httpx
import torch
import pydantic
from dataclasses import dataclass
import logging
import transformers
from .config import *

logger = logging.getLogger(__name__)

ChatCompletionMessages = list[dict[str, str]]

class GenerateArgs(pydantic.BaseModel):
    recursive: bool = pydantic.Field(default=True)
    max_tokens: int = pydantic.Field(strict=True, gt=0, default=256)
    seed: int | None = None
    context_window: int = pydantic.Field(default=8192, description="Context window size")

@dataclass
class ChatCompletionRequest:
    messages: ChatCompletionMessages
    args: GenerateArgs | None  = None


    def transform(self, fn: typing.Callable[[typing.Self], typing.Self]) -> typing.Self:
        return fn(self)


class Cluster(typing.Protocol):
    async def start(self):
        raise NotImplementedError("start is not implemented")

    def close(self) -> None:
        raise NotImplementedError("close is not implemented")

    async def new_worker(self, states: dict[str, torch.Tensor]) -> "Worker":
        raise NotImplementedError("new_worker is not implemented")

class Worker(typing.Protocol):
    def id(self) -> str:
        return ""

    async def close(self) -> None: 
        raise NotImplementedError("close is not implemented")


    async def chat_completion(self, messages: ChatCompletionMessages, args: GenerateArgs | None = None) -> str:
        raise NotImplementedError("chat_completion is not implemented")





@contextlib.asynccontextmanager
async def new_worker(cluster: Cluster, states: dict[str, torch.Tensor]):
    worker = await cluster.new_worker(states)
    try:
        yield worker
    except:
        import traceback
        logger.error("Error in worker: %s", traceback.format_exc())
        raise
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

async def _retry_post(client: httpx.AsyncClient, url: str, json_data: dict, http_config: HTTPConfig) -> httpx.Response:
    for attempt in range(http_config.retry_limit):
        try:
            resp = await client.post(url, json=json_data)
            resp.raise_for_status()
            return resp
        except httpx.RequestError as e:
            logger.warning(f"HTTP request error on attempt {attempt + 1}/{http_config.retry_limit}: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code / 100 == 4:
                logger.debug(f"Client error {e.response.status_code}: {e.response.text}, ignore retries.")
                raise 
            logger.warning(f"HTTP status error on attempt {attempt + 1}/{http_config.retry_limit}: {e}")

        backoff_time = http_config.retry_backoff * (2 ** attempt)
        backoff_time += random.uniform(0, http_config.retry_jitter)
        await asyncio.sleep(backoff_time)

    raise RuntimeError(f"Failed to POST to {url} after {http_config.retry_limit} attempts")


class VLLMWorker(Worker):
    def __init__(self, worker_id: str,  client: httpx.AsyncClient, addr: str, chat_template: "_ChatTemplate", http_config: HTTPConfig) -> None:
        self._worker_id = worker_id
        self._http_client = client
        self._addr = addr
        self._chat_template = chat_template
        self._http_config = http_config
    
    def id(self) -> str:
        return self._worker_id

    async def close(self) -> None:
        import traceback
        logger.debug("unloading lora adapter %s, stack %s", self._worker_id, "\n".join(traceback.format_stack()))
        resp = await self._http_client.post(
            f"{self._addr}/v1/unload_lora_adapter",
            json={
                "lora_name": self._worker_id,
            }
        )
        resp.raise_for_status()


    async def chat_completion(self, messages: list[dict[str, str]], args: GenerateArgs | None = None) -> str:
        if args is None:
            args = GenerateArgs()
        prompt: str = await self._chat_template.apply_chat_template(messages, add_generation_token=True)
        response: str = ""

        total_tokens: int = 0
        while True:
            if total_tokens + args.max_tokens > args.context_window:
                max_tokens = args.context_window - total_tokens
            else:
                max_tokens = args.max_tokens
            
            if max_tokens <= 0:
                return response

            request_json = {
                "model": self._worker_id,
                "prompt": prompt + response,
                "max_tokens": max_tokens,
            }
            if args.seed is not None:
                request_json["seed"] = args.seed
            try:
                resp = await _retry_post(self._http_client, f"{self._addr}/v1/completions", request_json, self._http_config)
            except httpx.HTTPStatusError as e:
                if e.response.status_code / 100 == 4:
                    # 4xx is client error
                    logger.debug("ignore request vllm error %d: %s", e.response.status_code,  e.response.text)
                    return response
                raise 
            response_json = resp.json()

            c0 = response_json["choices"][0]
            response += c0["text"]

            if c0["finish_reason"] != "length":
                return response

            total_tokens: int = response_json["usage"]["total_tokens"]


class VLLMCluster(Cluster):
    def __init__(self, config: VLLMClusterConfig, base_cfg: "Config"):
        self._config = config
        self._base_config = base_cfg
        self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(read=1800, connect=60, write=60, pool=5),
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
            http_config=self._base_config.http,
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
            http_config=self._base_config.http,
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
