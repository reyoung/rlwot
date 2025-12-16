
import pydantic
from typing import List, Annotated, Literal
import typing
import os

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

class HTTPConfig(pydantic.BaseModel):
    retry_limit: int = pydantic.Field(default=3, description="Retry limit")
    retry_backoff: float = pydantic.Field(default=0.5, description="Retry backoff factor")
    retry_jitter: float = pydantic.Field(default=0.1, description="Retry jitter")


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

    http: HTTPConfig = pydantic.Field(default=HTTPConfig(), description="HTTP configuration")

    def model_post_init(self, __context) -> None:
        """Create the save directory if it doesn't exist."""
        os.makedirs(self.save_dir, exist_ok=True)