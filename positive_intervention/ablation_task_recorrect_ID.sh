#!/bin/bash

#for model in "Qwen2.5-VL-3B-Instruct" "Qwen2.5-VL-7B-Instruct" "InternVL3-2B" "InternVL3-8B" "gemma-3n-e4b-it" "gemma-3n-e2b-it"
for model in "Qwen2.5-VL-3B-Instruct"
do
    for function_name in "Math Reasoning" "Vision Knowledge Recall" "Language Knowledge Recall" "Semantic Understanding" "Low-Level Vision Reception" "Inference" "High-Level Vision Reception" "Decision-Making"
    do
    python ablation_task_recorrect_all_ID.py \
    --model_name "$model" \
    --function_name "$function_name"
    done
done