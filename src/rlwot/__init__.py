import logging
import os
from typing import List, Optional

import torch
import transformers
import typer

logger = logging.getLogger(__name__)


def _name_in_target_modules(name: str, targets: list[str]) -> bool:
    return any(target in name for target in targets)


def generate_default_lora_model(
    base_model: str, target_modules: List[str], lora_r: int, save_dir: str
) -> dict[str, torch.Tensor]:
    """Generate a default LoRA model configuration.

    Args:
        base_model: Base model name
        target_modules: Target modules to apply lora
        lora_r: Rank of lora
        save_dir: Directory to save the LoRA model

    Returns:
        Dictionary containing LoRA tensors
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(base_model)
    res = {}

    for name, param in model.named_parameters():
        if not _name_in_target_modules(name, target_modules):
            continue

        assert param.dim() == 2, f"{name} is not a linear layer"

        rows, cols = param.shape

        logger.debug(f"generating lora for {name} with shape {rows}x{cols}")
        assert name.endswith(".weight"), "name must end with .weight"

        # base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
        prefix = name[: -len(".weight")]

        lora_a = torch.zeros((rows, lora_r), dtype=torch.bfloat16)
        lora_b = torch.zeros((lora_r, cols), dtype=torch.bfloat16)

        res[f"base.{prefix}.lora_A.weight"] = lora_a
        res[f"base.{prefix}.lora_B.weight"] = lora_b

    # Save the LoRA tensors to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lora_tensors.pt")
    torch.save(res, save_path)
    logger.info(f"Saved LoRA tensors to {save_path}")

    return res


def parse_args():
    """Parse command line arguments using typer.

    This function creates a default config when called programmatically.
    When the script is run directly, typer will handle the CLI parsing.
    """
    # For programmatic use, return default values
    return {
        "base_model": "Qwen/Qwen3-0.6B",
        "target_modules": [
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "v_proj",
            "k_proj",
        ],
        "lora_r": 2,
        "save_dir": "ckpt",
    }


# Create the typer app
app = typer.Typer()


@app.command()
def cli_main(
    base_model: str = typer.Option("Qwen/Qwen3-0.6B", help="Base model name"),
    target_modules: Optional[List[str]] = typer.Option(
        None,
        help="Target modules to apply lora (space separated)",
    ),
    lora_r: int = typer.Option(2, help="Rank of lora"),
    save_dir: str = typer.Option("ckpt", help="Directory to save the LoRA model"),
) -> None:
    """Generate a default LoRA model configuration."""
    # Set default values if not provided
    if target_modules is None:
        target_modules = [
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "v_proj",
            "k_proj",
        ]

    logging.basicConfig(level=logging.DEBUG)
    generate_default_lora_model(
        base_model=base_model,
        target_modules=target_modules,
        lora_r=lora_r,
        save_dir=save_dir,
    )


def main() -> None:
    """Entry point for the script."""
    app()


if __name__ == "__main__":
    app()
