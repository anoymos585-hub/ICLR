#!/bin/bash

#for model in "Qwen2.5-VL-3B-Instruct" "Qwen2.5-VL-7B-Instruct" "InternVL3-2B" "InternVL3-8B" "gemma-3n-e4b-it" "gemma-3n-e2b-it"
for model in "Qwen2.5-VL-3B-Instruct"
do
    for dataset_name in "okvqa" "Math_Vista" "visulogic"
    do
    python ablation_task_recorrect_all_OOD.py \
    --model_name "$model" \
    --dataset_fine "$dataset_name"
    done
done