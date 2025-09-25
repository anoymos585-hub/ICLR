#!/bin/bash

#for model in "Qwen2.5-VL-3B-Instruct" "Qwen2.5-VL-7B-Instruct" "InternVL3-2B" "InternVL3-8B" "gemma-3n-e4b-it" "gemma-3n-e2b-it"
for model in "Qwen2.5-VL-3B-Instruct"
do
    for low_function_name in "Vision Knowledge Recall" "Language Knowledge Recall" "Semantic Understanding" "Low-Level Vision Reception" "High-Level Vision Reception"
    do  
        for high_function_name in "Math Reasoning" "Decision-Making" "Inference"
        do
        python ablation_topk_cot.py \
        --low_function_name "$low_function_name" \
        --high_function_name "$high_function_name" \
        --model_name "$model" \
        --use_layer_bias
        done
    done
done