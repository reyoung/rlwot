import argparse
from datetime import datetime
import gc
import os
import random
import shutil
import signal
import sys
import typing
import numpy as np
import torch
import ray
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Qwen/Qwen3-4B')
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--polulation_size", type=int, default=30)
    parser.add_argument("--num_engines", type=int, default=4)
    parser.add_argument("--cuda_devices", type=str, default='0,1,2,3')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_dir", type=str, default='./experiments')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.experiment_dir = os.path.join(args.experiment_dir, f"{args.model_name.replace('/', '---')}"
                                       "dapo_math", datetime.now().strftime('%Y%m%d_%H%M%S'))

    return args

def save_base_model(args: argparse.Namespace, model_save_dir: str):
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base_model_dir = os.path.join(model_save_dir, "base_model")
    if os.path.exists(base_model_dir):
        shutil.rmtree(base_model_dir)
    os.makedirs(base_model_dir, exist_ok=True)
    tokenizer.save_pretrained(base_model_dir)
    base_model.save_pretrained(base_model_dir)
    del base_model
    gc.collect()

def _map_dapo_math(example):
    return {
        "prompt": example["prompt"],
        "ground_truth": example["reward_model"]["ground_truth"],
        "index": example["extra_info"]["index"]
    }

def load_dataset():
    train_dataset: datasets.Dataset = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")["train"]
    train_dataset = train_dataset.map(_map_dapo_math)
    return train_dataset

def main():
    args = parse_args()
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)
    engines = []
    pgs = []

    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in pgs:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    model_save_dir = os.path.join(args.experiment_dir, "models")

    save_base_model(args, model_save_dir)

    dataset = load_dataset()



if __name__ == "__main__":
    main()
