#!/bin/bash

#for model in "Qwen2.5-VL-3B-Instruct" "Qwen2.5-VL-7B-Instruct" "InternVL3-2B" "InternVL3-8B" "gemma-3n-e4b-it" "gemma-3n-e2b-it"
for model in "Qwen2.5-VL-3B-Instruct"
do
    for function in "Vision Knowledge Recall" "Language Knowledge Recall" "Semantic Understanding" "Math Reasoning" "Low-Level Vision Reception" "Inference" "High-Level Vision Reception" "Decision-Making"
    do  
        for head in "randomk" "topk"
        do
        python ablation_topk.py \
        --function_name "$function" \
        --use_head "$head" \
        --model_name "$model" \
        --use_layer_bias \
        --output_dir "head_acc.csv" \
        --folder_dir "main_results"
        done
    done
done