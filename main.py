import argparse
from datetime import datetime
import gc
import os
import random
import shutil
import signal
import sys
import numpy as np
import torch
import ray
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
from dapo_utils import verify as dapo_verify
import logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--population_size", type=int, default=32)
    parser.add_argument("--num_engines", type=int, default=4)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment_dir", type=str, default=os.path.expanduser("~/experiments")
    )
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--epoch_size", type=int, default=32)


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.experiment_dir = os.path.join(
        args.experiment_dir,
        f"{args.model_name.replace('/', '---')}dapo_math",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.experiment_dir, "run.log")),
            logging.StreamHandler(sys.stderr),
        ],
    )

    return args


def save_base_model(args: argparse.Namespace, model_save_dir: str):
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.float16, device_map="cpu", trust_remote_code=True
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
    return tokenizer


def _map_dapo_math(example):
    return {
        "prompt": example["prompt"],
        "ground_truth": example["reward_model"]["ground_truth"],
        "index": example["extra_info"]["index"],
    }


def load_dataset():
    train_dataset: datasets.Dataset = datasets.load_dataset(
        "BytedTsinghua-SIA/DAPO-Math-17k"
    )["train"]
    train_dataset = train_dataset.map(_map_dapo_math)
    return train_dataset


class MyLLM(LLM):
    def __init__(self, *args, **kwargs):
        # Let Ray/PG determine the actual visible device in the actor
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


def launch_engines(args: argparse.Namespace, model_path: str, engines: list, pgs: list):
    assert len(pgs) == 0
    assert len(engines) == 0
    pgs.extend(
        (
            ray.util.placement_group(
                [
                    {
                        "GPU": 1,
                        "CPU": 0,
                    }
                ]
            )
            for _ in range(args.num_engines)
        )
    )
    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]
    engines.extend(
        (
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(
                MyLLM
            ).remote(
                model=args.model_name,
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                worker_extension_cls="worker_ext.WorkerExtension",
                dtype="float16",
                enable_prefix_caching=False,
                enforce_eager=False,
            )
            for strategy in strategies
        )
    )

    master_address = "127.0.0.1"
    master_port = 56789
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])

def batch_loader(dataset, batch_size):
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch.clear()
    
    if batch:
        yield batch

def generate(llm, prompts: list[str]):
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=4096,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle

def _postprocess_outputs(outputs, samples):
    rewards = []
    for output, sample in zip(outputs, samples):
        ground_truth:str = sample["ground_truth"]
        response = output.outputs[0].text
        ok, _ = dapo_verify(response, ground_truth)
        rewards.append(float(ok))
    return {
        "rewards": rewards,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
    }

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

    tokenizer = save_base_model(args, model_save_dir)

    dataset = load_dataset()

    launch_engines(args, model_save_dir, engines, pgs)
    n_rollouts = 0

    for iteration in range(args.num_iterations):
        for epoch_id, epoch in enumerate(batch_loader(dataset, args.epoch_size)):
            seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
            n_rollouts += len(seeds) + len(epoch)
            seeds_perf = {}

            seed_iter = iter(seeds)
            inflight = {}
            results_this_gen = []

            prompts = [tokenizer.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True) for sample in epoch]

            # Kick off an eval on each engine
            for eng_idx, llm in enumerate(engines):
                try:
                    seed = next(seed_iter)
                except StopIteration:
                    break
                # Add exploration noise
                ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(seed, args.sigma, False)))
                handle = generate(llm, prompts)
                inflight[handle] = {
                    "engine": llm,
                    "engine_idx": eng_idx,
                    "seed": seed,
                }
            
            while inflight:
                done, _ = ray.wait(list(inflight.keys()), num_returns=1)
                h = done[0]

                meta = inflight.pop(h)
                outputs = ray.get(h)
                metrics = _postprocess_outputs(outputs, epoch)
                seeds_perf[meta["seed"]] = metrics
                results_this_gen.append(
                    {
                        "seed": meta["seed"],
                        "metrics": metrics,
                    }
                )

                llm = meta["engine"]
                # Remove exploration noise
                ray.get(llm.collective_rpc.remote("restore_self_weights", args=(meta["seed"], args.sigma)))

                logger.info(f"received results from engine {meta['engine_idx']}, {results_this_gen[-1]}")
                # Schedule next seed on this engine
                try:
                    next_seed = next(seed_iter)
                except StopIteration:
                    continue
                    
                ray.get(llm.collective_rpc.remote("perturb_self_weights", args=(next_seed, args.sigma, False)))
                handle = generate(llm, prompts)
                inflight[handle] = {
                    "engine": llm,
                    "engine_idx": meta["engine_idx"],
                    "seed": next_seed,
                }


            all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
            mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
            std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
            min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
            max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0
            logger.info(f"iteration={iteration} epoch={epoch_id} n_rollouts={n_rollouts} Mean reward: {mean_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}")

            for k in seeds_perf:
                seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_reward) / (std_reward + 1e-8)

            per_seed_coeffs = [
                (seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
                for seed in seeds
            ]
            handles = []
            for seed, coeff in per_seed_coeffs:
                # Use sigma_or_scale=1.0 so the applied scale is `coeff`
                handles.append(engines[0].collective_rpc.remote("perturb_self_weights", args=(seed, coeff, False)))
            ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])

    cleanup()


if __name__ == "__main__":
    main()
