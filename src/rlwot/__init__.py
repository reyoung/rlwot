import argparse
import contextlib
import logging
import os
import subprocess
import time
import typing
from pathlib import Path
from typing import Annotated, List, Literal

import httpx
import pydantic
import torch
import transformers
import yaml

logger = logging.getLogger(__name__)


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

    def as_args(self) -> typing.Generator[str, None, None]:
        yield "--port"
        yield str(self.port)


ClusterConfig = Annotated[
    typing.Union[StandaloneVLLMClusterConfig],
    pydantic.Field(discriminator="type"),
]


class Config(pydantic.BaseModel):
    """Configuration for LoRA model generation."""

    base_model: str = pydantic.Field(
        default="Qwen/Qwen3-0.6B", description="Base model name"
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
        default=2, description="Rank of LoRA"
    )

    save_dir: str = pydantic.Field(
        default="ckpt", description="Directory to save the LoRA model"
    )

    cluster_config: ClusterConfig = pydantic.Field(
        default=StandaloneVLLMClusterConfig(), description="Cluster configuration"
    )

    class Config:
        extra = "forbid"  # Forbid extra fields in the config

    def model_post_init(self, __context) -> None:
        """Create the save directory if it doesn't exist."""
        os.makedirs(self.save_dir, exist_ok=True)


class Cluster(typing.Protocol):
    def start(self, model_path: str) -> None: ...

    def close(self) -> None: ...


class StandaloneVLLMCluster(Cluster):
    def __init__(self, config: StandaloneVLLMClusterConfig):
        self._config = config
        self._vllm_process: subprocess.Popen | None = None
        self._http_client = httpx.Client()

    def start(self, model_path: str) -> None:
        args = ["vllm", "serve", model_path]
        args.extend(self._config.as_args())
        logger.info(f"Starting VLLM server with args: {args}")
        self._vllm_process = subprocess.Popen(args)

        # Waiting for the server is started
        try:
            self._wait_for_server()
        except:
            self._stop(force=True)
            raise

    def _wait_for_server(self):
        begin = time.time()
        while time.time() - begin < self._config.start_timeout_in_sec:
            try:
                self._http_client.get(f"http://localhost:{self._config.port}/health")
                break
            except httpx.RequestError:
                time.sleep(0.1)
        else:
            raise TimeoutError(
                f"VLLM server did not start within {self._config.start_timeout_in_sec} seconds"
            )

    def _stop(self, force=False):
        if self._vllm_process is not None:
            if not force:
                self._vllm_process.terminate()
            else:
                self._vllm_process.kill()
            self._vllm_process.wait()
            self._vllm_process = None

    def close(self):
        self._stop()


@contextlib.contextmanager
def connect_cluster(config: ClusterConfig, base_model: str):
    if isinstance(config, StandaloneVLLMClusterConfig):
        cluster = StandaloneVLLMCluster(config)
    else:
        raise ValueError(f"Unsupported cluster configuration: {config}")

    cluster.start(base_model)
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


def main() -> None:
    """Main entry point for script."""
    config = parse_args()
    logging.basicConfig(level=logging.DEBUG)
    with connect_cluster(config.cluster_config, config.base_model) as cluster:
        generate_default_lora_model(config)


if __name__ == "__main__":
    main()
