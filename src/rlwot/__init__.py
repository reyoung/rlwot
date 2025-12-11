import argparse
import logging
import os

import torch
import transformers

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        help="Target modules to apply lora",
        default=[
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "v_proj",
            "k_proj",
        ],
    )

    parser.add_argument("--lora-r", type=int, help="Rank of lora", default=2)

    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save the LoRA model",
        default="ckpt",
    )

    args = parser.parse_args()

    # post validate args

    os.makedirs(args.save_dir, exist_ok=True)

    return args


def _name_in_target_modules(name: str, targets: list[str]) -> bool:
    return any(target in name for target in targets)


# 1024x3072
def generate_default_lora_model(args) -> dict[str, torch.Tensor]:
    model = transformers.AutoModelForCausalLM.from_pretrained(args.base_model)
    targets = args.target_modules
    res = {}
    for name, param in model.named_parameters():
        if not _name_in_target_modules(name, targets):
            continue

        assert param.dim() == 2, f"{name} is not a linear layer"

        rows, cols = param.shape

        logger.debug(f"generating lora for {name} with shape {rows}x{cols}")
        assert name.endswith(".weight"), "name must end with .weight"

        # base_model.model.model.layers.0.mlp.down_proj.lora_A.weight

        prefix = name[: -len(".weight")]

        lora_a = torch.zeros((rows, args.lora_r), dtype=torch.bfloat16)
        lora_b = torch.zeros((args.lora_r, cols), dtype=torch.bfloat16)

        res[f"base.{prefix}.lora_A.weight"] = lora_a
        res[f"base.{prefix}.lora_B.weight"] = lora_b

    return res


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG)
    generate_default_lora_model(args)
