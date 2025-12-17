# RLWOT

Reinforcement Learning with Optimization Techniques

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Usage

After installation, you can run the command:

```bash
rlwot_es_naive --model_name Qwen/Qwen3-4B --sigma 0.001 --alpha 0.01 --population_size 32 --num_engines 4 --cuda_devices 0,1,2,3
```

## Command Line Options

- `--model_name`: Model name (default: Qwen/Qwen3-4B)
- `--sigma`: Noise scale (default: 0.001)
- `--alpha`: Learning rate (default: 0.01)
- `--population_size`: Population size for ES (default: 32)
- `--num_engines`: Number of vLLM engines (default: 4)
- `--cuda_devices`: CUDA devices to use (default: 0,1,2,3)
- `--seed`: Random seed (default: 42)
- `--experiment_dir`: Experiment directory (default: ~/experiments)
- `--num_iterations`: Number of iterations (default: 1)
- `--epoch_size`: Epoch size (default: 32)

## Docker Usage

Build and run using Docker:

```bash
# Build the image
docker build -t rlwot:latest .

# Run with GPU support
docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/experiments:/root/experiments \
  rlwot:latest rlwot_es_naive \
  --model_name Qwen/Qwen3-4B \
  --sigma 0.001 \
  --alpha 0.01

# Or use docker-compose
docker-compose up
```

For more details, see [DOCKER.md](DOCKER.md).