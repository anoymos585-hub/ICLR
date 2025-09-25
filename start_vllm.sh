#!/bin/bash
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate vllm_new
# # Set initial parameters

CUDA_VISIBLE_DEVICES=1,2,3 nohup vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --gpu_memory_utilization 0.8 \
  --max_model_len 8192 \
  --host 0.0.0.0 \
  --port 30000 \
  > ./logs/vllm_server.log 2>&1 &