#!/bin/bash

pids=()

for i in {0..3}
do
    vllm serve Qwen/Qwen3-4B-Instruct-2507 \
        --enable-lora --served-model-name base \
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