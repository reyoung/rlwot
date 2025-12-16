#!/bin/bash

pids=()
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export MODEL_NAME=Qwen/Qwen3-4B
export VLLM_BATCH_INVARIANT=1 
export DP_SIZE=${DP_SIZE:-4}


for ((i=0; i<$DP_SIZE; i++))
do
    CUDA_VISIBLE_DEVICES=$i vllm serve Qwen/Qwen3-4B \
        --enable-lora \
        --served-model-name base \
        --max-lora-rank 8 --max-loras 10 \
        --uvicorn-log-level error \
        --port 5820$i --max-model-len 8192 &
    pids+=($!)
done

read -p "Press [Enter] key to stop the servers"

for pid in ${pids[*]}
do
    kill $pid
done

wait