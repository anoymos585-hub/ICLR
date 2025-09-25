#!/bin/bash
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate vllm_new


for model in "Qwen2.5-VL-3B-Instruct"
do
    for function in "Vision Knowledge Recall" "Language Knowledge Recall" "Semantic Understanding" "Math Reasoning" "Low-Level Vision Reception" "Inference" "High-Level Vision Reception" "Decision-Making"
    #for function in "Decision-Making"
    do   
        for head in "topk"
        do
        python ablation_topk_other_function.py \
        --function_name "$function" \
        --use_head "$head" \
        --model_name "$model" \
        --output_dir "head_acc.csv" \
        --folder_dir "other_function_results"
        done
    done
done