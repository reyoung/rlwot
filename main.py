import argparse
import os
import random
import signal
import sys
import numpy as np
import torch
import ray
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Qwen/Qwen3-4B')
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--polulation_size", type=int, default=30)
    parser.add_argument("--num_engines", type=int, default=4)
    parser.add_argument("--cuda_devices", type=str, default='0,1,2,3')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    return args

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

if __name__ == "__main__":
    main()
