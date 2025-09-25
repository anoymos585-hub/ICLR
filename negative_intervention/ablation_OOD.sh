#!/bin/bash

#for model in "Qwen2.5-VL-7B-Instruct" "InternVL3-2B" "InternVL3-8B" "gemma-3n-e4b-it" "gemma-3n-e2b-it"
for model in "Qwen2.5-VL-3B-Instruct"
do
    for dataset_name in "../data/okvqa_dataset.json" "../data/Clevr_Math_test.json"
    do
        for head in "randomk" "topk"
        do
        python ablation_topk_OOD.py \
        --use_head "$head" \
        --model_name "$model" \
        --use_layer_bias \
        --dataset_name "$dataset_name" \
        --output_dir "head_acc.csv" \
        --folder_dir "OOD_results"
        done
    done
done