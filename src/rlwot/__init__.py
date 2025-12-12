import argparse
import logging
import os
from pathlib import Path
from typing import List

import pydantic
import torch
import transformers
import yaml

logger = logging.getLogger(__name__)


class LoRAConfig(pydantic.BaseModel):
    """Configuration for LoRA model generation."""

    base_model: str = "Qwen/Qwen3-0.6B"
    target_modules: List[str] = [
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "q_proj",
        "v_proj",
        "k_proj",
    ]
    lora_r: int = 2
    save_dir: str = "ckpt"

    def model_post_init(self, __context):
        """Create the save directory if it doesn't exist."""
        os.makedirs(self.save_dir, exist_ok=True)


def parse_args() -> LoRAConfig:
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

        return LoRAConfig(**config_dict)
    else:
        return LoRAConfig()


def _name_in_target_modules(name: str, targets: List[str]) -> bool:
    return any(target in name for target in targets)


def generate_default_lora_model(config: LoRAConfig) -> dict[str, torch.Tensor]:
    """Generate a default LoRA model configuration.

    Args:
        config: LoRAConfig object containing model parameters

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
    """Main entry point for the script."""
    config = parse_args()
    logging.basicConfig(level=logging.DEBUG)
    generate_default_lora_model(config)


if __name__ == "__main__":
    main()
