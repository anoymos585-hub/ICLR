import sys
import os
import argparse
from tqdm import tqdm
# from lib.utils.get_model import get_model, get_tokenizer
# from lib.utils.custommodel import CustomLlamaModelForCausalLM
# from lib.utils.format import get_time_str, set_seed
import logging
import os
import time
import torch
import gc
import json
# from accelerate import Accelerator
# import accelerate
import copy
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
# from lib.SHIPS.pd_diff import kl_divergence
# from lib.SHIPS.ships_utils import sort_ships_dict
# from lib.SHIPS.get_ships import SHIPS
from collections import defaultdict
import pickle
import math
import numpy as np
import random
from utils import cot_prompt, kl_scoring, simi_scoring, get_intervation_result, emb_similarity, cot_prompt_OOD, simi_scoring_intervention
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from interveners import wrapper, Collector, ITI_Intervener, head_Intervener
import pyvene as pv
from sentence_transformers import SentenceTransformer
import csv
from datasets import load_dataset
import re
from einops import rearrange
import evaluate
from collections import Counter
from openai import OpenAI


gpt_client = OpenAI(
    api_key='EMPTY',
    base_url=f'http://0.0.0.0:30000/v1',
)
gpt_model = gpt_client.models.list().data[0].id
print(f'model: {gpt_model}')

gpt_prompt = """
    You are a linguistic expert. You are give a predicted answer and ground truth answer. Determine whether the predict answer and ground truth answer have the same or similar meaning, return True. If most of the concepts are related or belong to a similar context/theme, treat them as a match even if one word is slightly different, more general, or more specific. Otherwise, return False

    Predict: {predicted}
    Ground Rruth: {ground_truth}  

    Answer with only one word: True or False.
"""

HF_NAMES = {
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_deepseek': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'llama3.1_8B_instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3.2_3B_instruct': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'qwen3-8B': 'Qwen/Qwen3-8B',
    'qwen3-1.7B': 'Qwen/Qwen3-1.7B',
    'qwen3-4B': 'Qwen/Qwen3-4B',
    'gemma3-12B': 'google/gemma-3-12b-it',
    'gemma3-4B': 'google/gemma-3-4b-it',
    'phi4-3.8B': "microsoft/Phi-4-mini-instruct",
    'internlm2-1.8B': 'internlm/internlm2-1_8b',
    'yi-1.5-6B': '01-ai/Yi-1.5-6B-Chat',
    'yi-1.5-9B': '01-ai/Yi-1.5-9B-Chat',
    'Qwen2.5-VL-3B-Instruct': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'Qwen2.5-VL-7B-Instruct': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'InternVL3-2B': 'OpenGVLab/InternVL3-2B-hf',
    'InternVL3-8B': 'OpenGVLab/InternVL3-8B-hf',
    'deepseek-vl2-small': 'deepseek-ai/deepseek-vl2-small',
    'deepseek-vl2-tiny': 'deepseek-ai/deepseek-vl2-tiny',
    'paligemma2-3b-pt-448': 'google/paligemma2-3b-pt-448',
    'paligemma2-10b-pt-448': 'google/paligemma2-10b-pt-448',
    'gemma-3n-e2b-it': 'google/gemma-3n-e2b-it',
    'gemma-3n-e4b-it': 'google/gemma-3n-e4b-it'
}   

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head
# # 定义干预函数
def run_intervention(inter_heads_idx, com_directions, args):

    pv_config = []
    for layer_idx, heads in inter_heads_idx.items():
        direction = torch.zeros(head_dim * num_heads).to("cpu")
        for head in heads:
            dir = torch.tensor(com_directions[layer_head_to_flattened_idx(layer_idx, head, num_heads)], dtype=torch.float32).to("cpu")
            dir = dir / torch.norm(dir)
            activations = torch.tensor(tuning_activations[:,layer_idx,head,:], dtype=torch.float32).to("cpu") # batch x 128
            proj_vals = activations @ dir.T
            proj_val_std = torch.std(proj_vals)
            direction[head * head_dim: (head + 1) * head_dim] = dir * proj_val_std
        # all_proj_vals = []
        # for head in heads:
        #     dir = torch.tensor(com_directions[layer_head_to_flattened_idx(layer_idx, head, num_heads)], dtype=torch.float32).to("cpu")
        #     dir = dir / torch.norm(dir)
        #     activations = torch.tensor(tuning_activations[:, layer_idx, head, :], dtype=torch.float32).to("cpu")
        #     proj_vals = activations @ dir.T
        #     all_proj_vals.append(proj_vals)

        # # 全局尺度
        # global_scale = torch.mean(torch.abs(torch.cat(all_proj_vals)))

        # for head in heads:
        #     dir = torch.tensor(com_directions[layer_head_to_flattened_idx(layer_idx, head, num_heads)], dtype=torch.float32).to("cpu")
        #     dir = dir / torch.norm(dir)
        #     direction[head * head_dim: (head + 1) * head_dim] = dir * global_scale
        intervener = ITI_Intervener(direction, args.alpha, args.sign) #head=-1 to collect all head activations, multiplier doens't matter
        # interveners.append(intervener)
        intervener.layer_idx = layer_idx  # 注意：为了让 __call__ 能访问当前层索引 
        pv_config.append({
            "component": f"model.language_model.layers[{layer_idx}].self_attn.o_proj.input",
            "intervention": wrapper(intervener),
        })
    intervened_model = pv.IntervenableModel(pv_config, model)
    
    return intervened_model

def run_intervention_all(inter_heads_idx, inter_heads_functions, sorted_stats, functions_com_directions, args):

    pv_config = []
    for layer_idx, heads in inter_heads_idx.items():
        direction = torch.zeros(head_dim * num_heads).to("cpu")
        for head in heads:
            # dir = torch.tensor(com_directions[layer_head_to_flattened_idx(layer_idx, head, num_heads)], dtype=torch.float32).to("cpu")
            idx = layer_head_to_flattened_idx(layer_idx, head, num_heads)
            functions = inter_heads_functions[idx]
            dir = torch.zeros(head_dim).to("cpu")
            for function in functions:
                dir = dir + torch.tensor(functions_com_directions[function][idx], dtype=torch.float32).to("cpu")
            dir = dir / len(functions)
            dir = dir / torch.norm(dir)
            activations = torch.tensor(tuning_activations[:,layer_idx,head,:], dtype=torch.float32).to("cpu") # batch x 128
            proj_vals = activations @ dir.T
            proj_val_std = torch.std(proj_vals)
            direction[head * head_dim: (head + 1) * head_dim] = dir * proj_val_std
        
        intervener = ITI_Intervener(direction, args.alpha, args.sign) #head=-1 to collect all head activations, multiplier doens't matter
        # interveners.append(intervener)
        intervener.layer_idx = layer_idx  # 注意：为了让 __call__ 能访问当前层索引 
        pv_config.append({
            "component": f"model.language_model.layers[{layer_idx}].self_attn.o_proj.input",
            "intervention": wrapper(intervener),
        })
    intervened_model = pv.IntervenableModel(pv_config, model)
    
    return intervened_model

def get_intervention_heads(layer_idx, head_idx, num_layers, num_heads):
    """
    获取指定层和头的干预索引
    """
    # 这里假设每个头的维度是相同的
    return {layer_idx: [head_idx]}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 额外，如果你用 Huggingface transformers
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass

def get_topk_intervention_heads(head_features):
    """
    获取指定层和头的干预索引
    :param head_features: List[int], 每个头的特征索引
    :param num_heads: int, 每层的头数
    :return: dict, 按层索引分组的头索引
    """
    inter_heads_idx = {}
    for feature in head_features:
        if type(feature) != int:
            feature = feature[0]
        layer_idx = feature // num_heads
        head_idx = feature % num_heads
        if layer_idx not in inter_heads_idx:
            inter_heads_idx[layer_idx] = [head_idx]
        else:
            inter_heads_idx[layer_idx].append(head_idx)
    
    return inter_heads_idx

def get_last_intervention_heads(head_features):
    save_heads_idx = {}
    inter_heads_idx = {}
    for feature in head_features:
        if type(feature) != int:
            feature = feature[0]
        layer_idx = feature // num_heads
        head_idx = feature % num_heads
        if layer_idx not in save_heads_idx:
            save_heads_idx[layer_idx] = [head_idx]
        else:
            save_heads_idx[layer_idx].append(head_idx)

    for i in range(num_layers):
        inter_heads_idx[i] = list(range(num_heads))
        if i in save_heads_idx:
            for head in save_heads_idx[i]:
                inter_heads_idx[i].remove(head)
    return inter_heads_idx

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj
    
system_role = """
You are an expert in analytical and logical reasoning. Your task is to answer the question.
"""

prompt = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
    - Think through the problem step by step.
    - Provide the final answer along with a brief explanation.
    - Be concise and avoid unnecessary details.

Only output your final answer using this format:
[
  {{
    "answer": "<Your final answer here>",
    "explanation": "<Your explanation here>"
  }}
]

Your answer:
"""

llama_prompt = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
- Think through the problem carefully and logically.
- Provide a step-by-step explanation leading to the final answer.
- Be concise and avoid unnecessary information.    

Output format:
[
  {{
    "explanation": "<Your explanation here>",
    "final answer": "<Your final answer here>"
  }}
]

Your answer:
"""    

other_prompt = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
- Think through the problem step by step to provide the explanation and the final answer.
- Be concise and avoid unnecessary information.   
- Your output must strictly follow the format shown below. 

Output format:
[
  {{
    "explanation": "<Your step-by-step thinking process here>",
    "final answer": "<Your final answer here>"
  }}
]

Your answer:
"""

gpt_prompt_onlyanswer = """
You are given a reference answer and a predicted answer to a question.

This is the reference answer:
<reference_answer>
{reference_answer}
</reference_answer>

This is the predicted answer:
<prediction_answer>
{prediction_answer}
</prediction_answer>

Instructions:
1. Return `"True"` if the prediction answer is consistent to the reference answer. Otherwise, return `"False"`.
2. Provide a **confidence score** between `0` and `1` based on how certain you are.

Output format:
[
{{
  "correct": <True|False>,
  "confidence": <float between 0 and 1>
}}
]
Your answer:
"""

def get_output(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_only_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()
    output_answer = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
    answer = ''
    explanation = ''
    match = re.search(r'"answer"\s*:\s*["\']?(.*?)["\']?,', output_answer)
    if match:
        answer = match.group(1)
        print("Answer:", answer)
    match = re.search(r'"explanation"\s*:\s*"(.+?)"\s*}', output_answer, re.DOTALL)
    if match:
        explanation = match.group(1).strip()
        print("Explanation:", explanation)
    else:
        print("No explanation found.")
    # match = re.search(r'\[\s*{\s*"answer":.*?"explanation":.*?}\s*]', output_answer, re.DOTALL)
    # if match:
    #     json_part = match.group()
    #     data = json.loads(json_part)
    #     answer = data[0]['answer']
    #     explanation = data[0]['explanation']
    #     print("Answer:", answer)
    #     print("Explanation:", explanation)
    # else:
    #     print("No valid JSON found.")
    # data = json.loads(output_answer)
    # answer = data[0]['answer']
    # explanation = data[0]['explanation']
    return answer, explanation

def input_prompt(dataset):
    all_prompts = []
    
    k = 0
    for i in range(len(dataset)):
        # question = dataset[i][0]['question']
        # answer = dataset[i][0]['answer']
        ref_answer = dataset[i]['answer']
        pre_answer = dataset[i]['response']["answer"]

        # format prompt
        prompt_text = gpt_prompt_onlyanswer.format(
            reference_answer=ref_answer,
            prediction_answer=pre_answer
        )
        k += 1

        all_prompts.append(prompt_text)

    return all_prompts

def get_result_from_output(outputs):
    answer_list = []
    confidence_list = []
    
    for i, output in enumerate(outputs):
        # output = re.sub(r'"True"', 'True', output[0][0])
        # output = re.sub(r'"False"', 'False', output)
        output = output[0][0].replace("True", "true").replace("False", "false")
        matches = re.findall(r'\[\s*{[^}]+}\s*\]', output)
        if matches:
            output_json_block = matches[-1].strip()
            print("Extracted JSON answer block:", output_json_block)

            try:
                # parsed = safe_json_parse(output_json_block)
                parsed = json.loads(output_json_block)
                answer, confidence = parsed[0]["correct"], parsed[0]["confidence"]  # This will return '"$80"'
                print("Final extracted answer (with quotes):", answer)
            except:
                answer_match = re.search(r'"correct":\s*"([^"]+)"', output)
                confidence_match = re.search(r'"confidence":\s*([\d.]+)', output)
                
                # 如果匹配到值则返回
                if answer_match and confidence_match:
                    answer = answer_match.group(1)
                    confidence = float(confidence_match.group(1))

                else:
                    answer = ""
                    confidence = 0
        
        answer_list.append(answer)
        confidence_list.append(confidence)
        
        # answer_list2.append(answer2)
        # confidence_list2.append(confidence2)
        
        print(f"Answer: {answer}, Confidence: {confidence}")
    
    return answer_list, confidence_list

def replace_nan_in_list(data_list):
    for i in range(len(data_list)):
        data_list[i] = np.nan_to_num(data_list[i], nan=0.0, posinf=1e4, neginf=-1e4)
        if np.isnan(data_list[i]).any() or np.isinf(data_list[i]).any():
            print("aa")
    return data_list

def load_data(model_name, layer_num, heads_num, dim, position, head_num=-1, mode="test"):
    if ("7B" in model_name or "8B" in model_name or "4b" in model_name) and mode == "train":
        with open(f'../head_results/{model_name}_train_1000_head_wise_train_0.pkl', 'rb') as f:
            train_data_0 = pickle.load(f)
        print("load 0...")
        with open(f'../head_results/{model_name}_train_1000_head_wise_train_1.pkl', 'rb') as f:
            train_data_1 = pickle.load(f)
        print("load 1...")
        with open(f'../head_results/{model_name}_train_1000_head_wise_train_2.pkl', 'rb') as f:
            train_data_2 = pickle.load(f)
        print("load 2...")
        with open(f'../head_results/{model_name}_train_1000_head_wise_train_3.pkl', 'rb') as f:
            train_data_3 = pickle.load(f)
        print("load 3...")
        data = train_data_0 + train_data_1 + train_data_2 + train_data_3
    else:
        with open(f'../head_results/{model_name}_{mode}_1000_head_wise_{mode}.pkl', 'rb') as f:
            data = pickle.load(f)
    data = replace_nan_in_list(data)

    with open(f'../head_results/{model_name}_{mode}_1000_topk_position_{mode}.pkl', "rb") as f:
        topk_position = pickle.load(f) 

    features = []
    for i in range(len(data)):
        token_num = data[i].shape[1]
        if args.token_use == "topk":
            token_select = [x for x in topk_position[i] if 0 <= x < token_num]
        if token_select == []:
            print("topk position empty!!")
            token_select = [0]
        reshaped = data[i].reshape(layer_num, token_num, heads_num, dim)
        reshaped = reshaped[:, token_select, :, :]
        
        if position in ["topk", "full"]:
            reshaped = np.mean(reshaped, axis=1)


        reshaped = reshaped.reshape(layer_num * heads_num, dim)

        features.append(reshaped if head_num == -1 else reshaped[head_num])
    if np.isnan(features).any() or np.isinf(features).any():
        print("nan exits!!")
    return features

def load_data_fine(model_name, dataset_name, layer_num, heads_num, dim, position, head_num=-1):
    if dataset_name == "vqav2":
        dataset_name = "okvqa"
    if ("7B" in model_name or "8B" in model_name or "4b" in model_name):
        with open(f'../inter_results/{model_name}_{dataset_name}_head_wise_train_0.pkl', 'rb') as f:
            train_data_0 = pickle.load(f)
        print("load 0...")
        with open(f'../inter_results/{model_name}_{dataset_name}_head_wise_train_1.pkl', 'rb') as f:
            train_data_1 = pickle.load(f)
        print("load 1...")
        with open(f'../inter_results/{model_name}_{dataset_name}_head_wise_train_2.pkl', 'rb') as f:
            train_data_2 = pickle.load(f)
        print("load 2...")
        with open(f'../inter_results/{model_name}_{dataset_name}_head_wise_train_3.pkl', 'rb') as f:
            train_data_3 = pickle.load(f)
        print("load 3...")
        data = train_data_0 + train_data_1 + train_data_2 + train_data_3
    else:
        with open(f'../inter_results/{model_name}_{dataset_name}_head_wise_train.pkl', 'rb') as f:
            data = pickle.load(f)

    with open(f'../inter_results/{model_name}_{dataset_name}_topk_position_train.pkl', "rb") as f:
        topk_position = pickle.load(f) 
    data = replace_nan_in_list(data)
    features = []
    for i in range(len(data)):
        token_num = data[i].shape[1]
        if args.token_use == "topk":
            token_select = [x for x in topk_position[i] if 0 <= x < token_num]
        if token_select == []:
            print("topk position empty!!")
            token_select = [0]
        reshaped = data[i].reshape(layer_num, token_num, heads_num, dim)
        reshaped = reshaped[:, token_select, :, :]

        if position in ["topk", "full"]:
            reshaped = np.mean(reshaped, axis=1)


        reshaped = reshaped.reshape(layer_num * heads_num, dim)

        features.append(reshaped if head_num == -1 else reshaped[head_num])
    return features

def get_com_directions(num_layers, num_heads, usable_idxs, head_wise_activations, answers): 
    labels = []
    for i in usable_idxs:
        labels.append(answers[i][0]['label'])
    str_indices = [i for i, v in enumerate(labels) if isinstance(v, str)]
    for i in str_indices:
        if labels[i] == "True" or labels[i] == "true":
            labels[i] = True
        elif labels[i] == "False" or labels[i] == "false":
            labels[i] = False

    usable_labels = [int(l) for l in labels]
    
    com_directions = []
    # head_wise_activations = head_wise_activations.reshape(num_layers, num_heads, -1)
    for layer in tqdm(range(num_layers), desc="get_com_directions"): 
        for head in range(num_heads): 
            usable_head_wise_activations = [head_wise_activations[i].reshape(num_layers, num_heads, -1)[layer,head,:] for i in usable_idxs]
            usable_head_wise_activations = np.stack(usable_head_wise_activations)
            labels_1 = np.where(np.array(usable_labels) == 1)[0]
            labels_0 = np.where(np.array(usable_labels) == 0)[0]
            true_mass_mean = np.mean(usable_head_wise_activations[labels_1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[labels_0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions

def get_functions_com_directions(num_layers, num_heads, train_idxs_functions, test_idxs_functions, train_head_wise_activations_all, test_head_wise_activations_all):
    functions_com_directions = {}
    for i in functions_names:
        train_idxs = train_idxs_functions[i]["idx"]
        test_idxs = test_idxs_functions[i]["idx"]
        train_labels = train_idxs_functions[i]["labels"]
        test_labels = test_idxs_functions[i]["labels"]
        if len(train_idxs) == 0 or len(test_idxs) == 0:
            continue
        train_head_wise_activations = [train_head_wise_activations_all[j].reshape(num_layers, num_heads, -1) for j in train_idxs]
        test_head_wise_activations = [test_head_wise_activations_all[j].reshape(num_layers, num_heads, -1) for j in test_idxs]
        functions_com_direction = []
        for layer in tqdm(range(num_layers), desc="get_functions_com_directions"): 
            for head in range(num_heads): 
                usable_train_head_wise_activations = [train_head_wise_activations[j][layer,head,:] for j in range(len(train_head_wise_activations))]
                usable_train_head_wise_activations = np.stack(usable_train_head_wise_activations)
                labels_1 = np.where(np.array(train_labels) == 1)[0]
                labels_0 = np.where(np.array(train_labels) == 0)[0]
                usable_test_head_wise_activations = [test_head_wise_activations[j][layer,head,:] for j in range(len(test_head_wise_activations))]
                usable_test_head_wise_activations = np.stack(usable_test_head_wise_activations)
                labels_1_test = np.where(np.array(test_labels) == 1)[0]
                labels_0_test = np.where(np.array(test_labels) == 0)[0]
                true_usable_head_wise_activations = np.concatenate((usable_train_head_wise_activations[labels_1], usable_test_head_wise_activations[labels_1_test]), axis=0)
                false_usable_head_wise_activations = np.concatenate((usable_train_head_wise_activations[labels_0], usable_test_head_wise_activations[labels_0_test]), axis=0)
                # true_mass_mean = np.mean(usable_train_head_wise_activations[labels_1], axis=0)
                # false_mass_mean = np.mean(usable_train_head_wise_activations[labels_0], axis=0)
                true_mass_mean = np.mean(true_usable_head_wise_activations, axis=0)
                false_mass_mean = np.mean(false_usable_head_wise_activations, axis=0)
                com_direction = true_mass_mean - false_mass_mean
                functions_com_direction.append(com_direction)
        functions_com_direction = np.array(functions_com_direction)
        functions_com_directions[i] = functions_com_direction
    return functions_com_directions


def scoring(base_prediction, reference_answer):
    predictions = [str(base_prediction)]
    references = [str(reference_answer)]
    results = {}

    # BLEU（支持多个参考）

    bleu_score = bleu.compute(predictions=predictions, references=references)
    results["bleu"] = bleu_score["bleu"]

    # ROUGE
    rouge_score = rouge.compute(predictions=predictions, references=references)
    results.update(rouge_score)  # rouge1, rouge2, rougeL, rougeLsum
    
    cosine_score = emb_similarity([predictions, references])
    results["cosine"] = cosine_score

    # COMET（需要源句）
    if args.use_comet:
        comet_score = comet.compute(
            predictions=predictions,
            references=references,
            sources=references
        )
        results["comet"] = comet_score["mean_score"]
    else:
        results["comet"] = "Skipped (no sources provided)"
    return results

def get_idxs_of_functions(dataset_name, model_name, functions_names):
    if dataset_name == "cot_qa":
        train_path = '../dataset/train_1000_final.json'
        train_data = json.load(open(train_path, "r"))
        test_path = '../dataset/test_1000_final.json'
        test_data = json.load(open(test_path, "r")) 
        train_label_path = "../head_results/output_" + model_name + "_" + "train_1000" + "_train" + "_with_gpt_label.json"
        train_data_label = json.load(open(train_label_path, "r"))
        test_label_path = "../head_results/output_" + model_name + "_" + "test_1000" + "_test" + "_with_gpt_label.json"
        test_data_label = json.load(open(test_label_path, "r"))
    train_functions = []
    for i in range(len(train_data)):
        generated = train_data[i]["generated"]
        for j in range(len(generated)):
            label = generated[j]["cognitive_skill"]
            train_functions.append(label)
    test_functions = []
    for i in range(len(test_data)):
        generated = test_data[i]["generated"]
        for j in range(len(generated)):
            label = generated[j]["cognitive_skill"]
            test_functions.append(label)
    train_labels = []
    for i in range(len(train_data_label)):
        train_labels.append(train_data_label[i][0]['label'])
    str_indices = [i for i, v in enumerate(train_labels) if isinstance(v, str)]
    for i in str_indices:
        if train_labels[i] == "True" or train_labels[i] == "true":
            train_labels[i] = True
        elif train_labels[i] == "False" or train_labels[i] == "false" or train_labels[i] == "":
            train_labels[i] = False
    train_labels = [int(l) for l in train_labels]
    
    test_labels = []
    for i in range(len(test_data_label)):
        test_labels.append(test_data_label[i][0]['label'])
    str_indices = [i for i, v in enumerate(test_labels) if isinstance(v, str)]
    for i in str_indices:
        if test_labels[i] == "True" or test_labels[i] == "true":
            test_labels[i] = True
        elif test_labels[i] == "False" or test_labels[i] == "false" or test_labels[i] == "":
            test_labels[i] = False
    test_labels = [int(l) for l in test_labels]
    train_function_list = defaultdict(list)
    test_function_list = defaultdict(list)
    for idx, category in enumerate(train_functions):
        train_function_list[category].append(idx)
    for idx, category in enumerate(test_functions):
        test_function_list[category].append(idx)

    train_idxs_functions = {}
    for i in range(len(functions_names)):
        if functions_names[i] in train_function_list:
            idx = train_function_list[functions_names[i]]
            labels = [train_labels[j] for j in idx]
            save_idx = {
            "idx": idx,
            "labels": labels
            }
            train_idxs_functions[functions_names[i]] = save_idx
        else:
            train_idxs_functions.append([])
    test_idxs_functions = {}
    for i in range(len(functions_names)):
        if functions_names[i] in test_function_list:
            idx = test_function_list[functions_names[i]]
            labels = [test_labels[j] for j in idx]
            save_idx ={"idx": idx, "labels": labels}
            test_idxs_functions[functions_names[i]] = save_idx
        else:
            test_idxs_functions.append([])
    
    return train_idxs_functions, test_idxs_functions


def extract_answer(output_text):
    # 提取最后的 "#### 数字" 作为模型的答案
    match = re.search(r"####\s*(-?[0-9,\.]+)", output_text)
    if match:
        return match.group(1).replace(",", "")
    return None

def extract_number(text):
    # 提取第一个整数或小数（支持负号）
    match = re.search(r'-?\d+(\.\d+)?', text)
    return float(match.group()) if match else None

def extract_explanations_and_answers(text):
    """
    提取文本中所有 JSON 格式的回答中的 explanation 和 final answer。
    如果没有完整的 JSON 块，则回退使用正则表达式提取字段。

    参数:
        text (str): 原始字符串，包含多个 JSON 段或字段。

    返回:
        List[Dict[str, str]]: 每个字典包含 'explanation' 和 'final_answer'。
    """
    results = []

    # 第一种方式：尝试从 JSON 块中提取
    json_blocks = re.findall(r'\[\s*\{.*?\}\s*\]', text, flags=re.DOTALL)
    
    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list) and isinstance(data[0], dict):
                explanation = str(data[0].get("explanation", "")).strip()
                final_answer = str(data[0].get("final answer", "")).strip()
                results.append({
                    "explanation": explanation,
                    "final_answer": final_answer
                })
        except Exception as e:
            print(f"解析失败: {e}")
            continue

    # 如果没有解析出任何 JSON 块，则退回正则匹配字段
    if not results:
        explanations = re.findall(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
        final_answers = re.findall(r'"final answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)

        for exp, ans in zip(explanations, final_answers):
            results.append({
                "explanation": exp.strip(),
                "final_answer": ans.strip()
            })

    return results

functions_names = ["Vision Knowledge Recall", "Language Knowledge Recall", "Semantic Understanding", "Math Reasoning", "Low-Level Vision Reception", "Inference", "High-Level Vision Reception", "Decision-Making"]

prompt_structure = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
    - Think through the problem step by step.
    - Provide the final answer along with a brief explanation.
    - Be concise and avoid unnecessary details.

Only output your final answer using this format:
[
  {{
    "answer": "<Your final answer here>",
    "explanation": "<Your explanation here>"
  }}
]

Your answer:
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='cot_qa')
    parser.add_argument('--dataset_fine', type=str, default='okvqa')
    parser.add_argument('--activations_dataset', type=str, default=None)
    parser.add_argument('--intervation_way', type=str, default='single')
    parser.add_argument('--recorrect_all', default=True, action='store_true')
    parser.add_argument('--folder_dir', type=str, default='pos_intervention')
    parser.add_argument('--alpha', type=float, default=1, help='alpha, intervention strength')
    parser.add_argument('--token_use', type=str, default="topk")
    parser.add_argument('--mask_ratio', type=int, default=30)
    parser.add_argument('--mask_ratio_all', type=float, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--llm_model_method', default = 'o4-mini', type = str)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--function_name', type=str, default='Math Reasoning')
    parser.add_argument('--use_bleu', default=True, action='store_true')
    parser.add_argument('--use_rouge', default=True, action='store_true')
    parser.add_argument('--use_cosine', default=True, action='store_true')
    parser.add_argument('--use_comet', default=True, action='store_true')
    parser.add_argument('--use_kl', default=False, action='store_true')
    parser.add_argument('--use_head', type=str, default="topk")
    parser.add_argument('--use_layer_bias', default=True, action="store_true")
    parser.add_argument('--output_dir', type=str, default="recorrect_head_acc.csv")
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--sign', type=str, default='plus')
    parser.add_argument('--head_ratio', type=float, default=0.1)
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    train_idxs_functions, test_idxs_functions = get_idxs_of_functions(args.dataset_name, args.model_name, functions_names)


    model_name_or_path = HF_NAMES[args.model_name]
    if "Qwen" in args.model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name_or_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True)
    elif "Intern" in args.model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True)
    elif "paligemma2" in args.model_name:
        print("load paligemma2")
        tokenizer = PaliGemmaProcessor.from_pretrained(model_name_or_path)
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True)
    device = "cuda"
  
    with open(f'../model_config.json', "r") as f:
        model_config = json.load(f)
    for config in model_config:
        if config["model_name"] == args.model_name:
            num_layers = config["layer_num"]
            num_heads = config["heads_num"]
            head_dim = config["dim"]
            break
    args.mask_ratio = math.ceil(args.head_ratio * num_layers * num_heads)
    #if "InternVL3-2B" in args.model_name:
    #args.mask_ratio = args.mask_ratio * 2 
        
    mask_config = {
        "scale_factor": 0.0001,
        "mask_type": "scale_mask",
    }
    

    dataset = json.load(open("../dataset/test_1000_final.json", "r")) 

    if args.dataset_fine == "okvqa":
        path = '../data/okvqa_dataset.json'
        dataset_fine = json.load(open(path, "r"))
    elif args.dataset_fine == "clevr":
        # dataset_fine = load_dataset("gsm8k", "main") 
        # dataset_fine = dataset_fine['test'][:10]     
        path = '../data/Clevr_Math_test.json' 
        dataset_fine = json.load(open(path, "r"))
    elif args.dataset_fine == "Math_Vision":
        # dataset_fine = load_dataset("gsm8k", "main") 
        # dataset_fine = dataset_fine['test'][:10]     
        path = '../data/Math_Vision_test.json' 
        dataset_fine = json.load(open(path, "r"))
    elif args.dataset_fine == "Math_Vista":
        # dataset_fine = load_dataset("gsm8k", "main") 
        # dataset_fine = dataset_fine['test'][:10]     
        path = '../data/Math_Vista_test.json' 
        dataset_fine = json.load(open(path, "r"))
    elif args.dataset_fine == "vqav2":
        # dataset_fine = load_dataset("gsm8k", "main") 
        # dataset_fine = dataset_fine['test'][:10]     
        path = '../data/vqav2.json' 
        dataset_fine = json.load(open(path, "r"))
    elif args.dataset_fine == "visulogic":
        # dataset_fine = load_dataset("gsm8k", "main") 
        # dataset_fine = dataset_fine['test'][:10]     
        path = '../data/visulogic_dataset.json' 
        dataset_fine = json.load(open(path, "r"))
    with open(f"../main_results/{args.model_name}/importance_{args.model_name}_{args.token_use}_indices_{args.use_layer_bias}.json", "r") as f:
        importances = json.load(f)
        
  
    tuning_activations = load_data_fine(args.model_name, args.dataset_fine, num_layers, num_heads, head_dim, args.token_use)
    tuning_activations = np.stack(tuning_activations)
    tuning_activations = rearrange(tuning_activations, 'b (l h) d -> b l h d', l=num_layers, h = num_heads)
    
    train_head_wise_activations = load_data(args.model_name, num_layers, num_heads, head_dim, args.token_use, mode="train")
    test_head_wise_activations = load_data(args.model_name, num_layers, num_heads, head_dim, args.token_use, mode="test")
    functions_com_direction = get_functions_com_directions(num_layers, num_heads, train_idxs_functions, test_idxs_functions, train_head_wise_activations, test_head_wise_activations)

    # tuning_activations = rearrange(train_head_wise_activations, 'b (l h) d -> b l h d', l=num_layers, h = num_heads)

    def get_intervened_model_for_each(functions_names, args):
        if args.intervation_way == "all":
            head_feature_list = {}
            for i in range(len(functions_names)):
                head_features = importances[functions_names[i]][:args.mask_ratio]
                #head_features = [feature[0] for feature in head_features]
                head_feature_list[functions_names[i]] = head_features
            index_to_keys = defaultdict(set)
            for key, idx_list in head_feature_list.items():
                for idx in idx_list:
                    index_to_keys[idx].add(key)

            # 按 index 出现在多个类别的情况统计，并排序
            # sorted_stats = sorted(index_to_keys.items(), key=lambda x: -len(x[1]))
            sorted_stats = [(k, v) for k, v in index_to_keys.items() if len(v) == 1]
            topk = int(args.mask_ratio_all*len(sorted_stats))
            head_features = [feature[0] for feature in sorted_stats[:topk]]   
            inter_heads_idx = get_topk_intervention_heads(head_features)     
            inter_heads_functions = {feature[0]: feature[1] for feature in sorted_stats[:topk]}
            
            # feature_counts = Counter(head_feature_list)
            # sorted_features = feature_counts.most_common()
            # head_features = [feature[0] for feature in sorted_features[:int(args.mask_ratio_all*len(sorted_features))]]
            # head_features = {k: v for k, v in feature_counts.items() if v > 3}
        
        elif args.intervation_way == "single":
            if args.use_head == "randomk":
                head_features = importances[args.function_name[0]][:args.mask_ratio]
                all_heads = list(range(num_layers * num_heads))
                all_heads = [h for h in all_heads if h not in head_features]
                head_features = random.sample(all_heads, args.mask_ratio)
            elif args.use_head == "lowk":
                head_features = importances[args.function_name[0]][-args.mask_ratio:]
            elif args.use_head == "topk":
                head_features = importances[args.function_name[0]][:args.mask_ratio]
            #print(head_features)
            # 遍历每层每个 head 进行干预    acc_head = {}
            # for i in tqdm(range(num_layers), desc="Layers"):
            #     for j in range(num_heads):
            #         inter_heads_idx = get_intervention_heads(i, j, num_layers, num_heads)
            if args.use_head in ["topk", "randomk", "lowk"]:
                inter_heads_idx = get_topk_intervention_heads(head_features)  # 这里假设你只想干预第 0 层的第 0 个头
            elif args.use_head == "last":
                inter_heads_idx = get_last_intervention_heads(head_features)

        # with open("head_results/output_" + args.model_name + "_" + args.dataset_name + "_" + args.mode + "_with_gpt_label.json", "r") as f:
        #     labels = json.load(f)
        # data_idxs = np.arange(len(dataset))
        # com_direction = get_com_directions(num_layers, num_heads, data_idxs, head_wise_activations, labels)
        if args.intervation_way == "all":
            # com_direction = get_com_directions(num_layers, num_heads, head_features, head_wise_activations, dataset_fine)
            intervened_model = run_intervention_all(inter_heads_idx, inter_heads_functions, sorted_stats, functions_com_direction, args)
        elif args.intervation_way == "single":
            com_direction = functions_com_direction[args.function_name[0]]
            intervened_model = run_intervention(inter_heads_idx, com_direction, args)
    
        # intervened_model = run_intervention(inter_heads_idx, com_direction)
        return intervened_model
   
        # 推理
    #get_intervened_model_for_each(["Math Reasoning"])
    scores_list = []
    base_scores_list = []
    generated_answers = []
    acc_head = {}
    acc_num = 0
    base_acc_num = 0
    adv_acc_num = 0    

    check_dataset = []
    check_dataset1 = json.load(open("../dataset/test_1000_final.json"))
    check_dataset2 = json.load(open("../dataset/train_1000_final.json"))
    for each in check_dataset1:
        check_dataset.append("../"+each["image_path"])
    for each in check_dataset2:
        check_dataset.append("../"+each["image_path"])
    prompts, answers, images = cot_prompt_OOD(dataset_fine, check_dataset, num=200)

    evaluate_model = SentenceTransformer('all-MiniLM-L6-v2')
    base_results = []
    adv_results = []  
    #prompts = [prompts[i] for i in dataset_idx]
    #dataset_fine = [dataset_fine[i] for i in dataset_idx]
    if "okvqa" in args.dataset_fine:
        args.function_name = ["High-Level Vision Reception"]
    # if "Math_Vision" in args.dataset_name:
    #     args.function_name = "Math Reasoning"
    if "clevr" in args.dataset_fine:
        args.function_name = ["Math Reasoning"]
    if "Math_Vision" in args.dataset_fine:
        args.function_name = ["Math Reasoning"]
    if "vqav2" in args.dataset_fine:
        args.function_name = ["High-Level Vision Reception"]
    if "Math_Vista" in args.dataset_fine:
        args.function_name = ["Math Reasoning"]
    if "visulogic" in args.dataset_fine:
        args.function_name = ["Decision-Making"]
    # dataset_fine["question"] = [dataset_fine["question"][i] for i in dataset_idx]
    # dataset_fine["answer"] = [dataset_fine["answer"][i] for i in dataset_idx]
    if not os.path.exists(os.path.join(f"{args.folder_dir}/{args.model_name}", args.function_name[0].replace(" ", "_"))):
        os.makedirs(os.path.join(f"{args.folder_dir}/{args.model_name}", args.function_name[0].replace(" ", "_")))
    for k in tqdm(range(len(prompts))):
        if "okvqa" in args.dataset_fine:
            prompts[k] = prompts[k] + " The answer should be short and concise."

        # prompt = prompt_structure.format(
        #     question=prompts[k]
        # )
        prompt = prompt_structure.format(
            system_role=system_role,
            question=prompts[k]
        )
        label = args.function_name
        answer = answers[k]
        image = images[k]
        #label = args.function_name
        # intervened_functions = [subqac['cognitive_skill'] for subqac in dataset_fine[k]["generated"]]
        #intervened_functions =["Semantic Understanding", "Math Calculation"]
        intervened_model = get_intervened_model_for_each(args.function_name, args)
        # if args.dataset_fine == "GSM8K":
        #     answer = extract_answer(dataset_fine[k]["answer"])
        #     explanation = dataset_fine[k]["answer"]

        #base_answer, adv_answer = get_intervation_result(prompt, tokenizer, intervened_model, device, args)
        base_answer, base_explanation, adv_answer, adv_explanation, scores, base_scores, kl_scores = simi_scoring_intervention(prompt, label, answer, image, args.model_name, tokenizer, intervened_model, device, args, evaluate_model)


        llm_judge_prompt = gpt_prompt.format(
                question=prompt[k],
                predicted=adv_answer,
                ground_truth=answer,
            )
        #print(llm_judge_prompt)
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': llm_judge_prompt}
        ]}]

        resp = gpt_client.chat.completions.create(model=gpt_model, messages=messages, max_tokens=512, temperature=0)
        response = resp.choices[0].message.content
        #print("LLM response: ", response)
        if "true" in response.lower():
            scores["llm"] = 1.0
        else:
            scores["llm"] = 0.0
        scores_list.append(scores)

        llm_judge_prompt = gpt_prompt.format(
                question=prompt[k],
                predicted=base_answer,
                ground_truth=answer,
            )
        #print(llm_judge_prompt)
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': llm_judge_prompt}
        ]}]

        resp = gpt_client.chat.completions.create(model=gpt_model, messages=messages, max_tokens=512, temperature=0)
        response = resp.choices[0].message.content
        #print("LLM response: ", response)
        if "true" in response.lower():
            base_scores["llm"] = 1.0
        else:
            base_scores["llm"] = 0.0
        base_scores_list.append(base_scores)
        if base_scores["llm"] == 1.0 and scores["llm"] == 0.0:
            with open(os.path.join(f"./{args.folder_dir}/{args.model_name}", args.function_name[0].replace(" ", "_"), args.function_name[0].replace(" ", "_") + "_" + args.use_head +str(args.use_layer_bias)+ "_adv_incorrect.jsonl"), "a") as f:
                # f.write(f"{args.use_head}: {adv_answer} || {base_answer} || {answer}\n")
                f.write(json.dumps({"head": args.use_head, "adv_answer": adv_answer, "adv_explanation": adv_explanation, "base_answer": base_answer, "base_explanation": base_explanation, "answer":answer}, ensure_ascii=False) + "\n")

        with open(os.path.join(f"./{args.folder_dir}/{args.model_name}", args.function_name[0].replace(" ", "_"), args.function_name[0].replace(" ", "_") + "_" + args.use_head +str(args.use_layer_bias)+ "_head_answer.jsonl"), "a") as f:
            # f.write(f"{args.use_head}: {adv_answer} || {base_answer} || {answer}\n")
            f.write(json.dumps({"head": args.use_head, "adv_answer": adv_answer, "adv_explanation": adv_explanation, "base_answer": base_answer, "base_explanation": base_explanation, "answer":answer}, ensure_ascii=False) + "\n")
        with open(os.path.join(f"./{args.folder_dir}/{args.model_name}", args.function_name[0].replace(" ", "_"), args.function_name[0].replace(" ", "_") + "_" + args.use_head+str(args.use_layer_bias) +"_head_score.txt"), "a") as f:
            f.write(f"{args.use_head}: {scores}\n")
            f.write(f"{args.use_head}: {base_scores}\n")
        torch.cuda.empty_cache()

    if args.use_bleu:
        scores_bleu = [score["bleu"] for score in scores_list]
        score_bleu = np.mean(scores_bleu)
        print(f"BLEU: {score_bleu:.6f}")
    if args.use_rouge:
        scores_rouge1 = [score["rouge1"] for score in scores_list]
        score_rouge1 = np.mean(scores_rouge1)
        scores_rougeL = [score["rougeL"] for score in scores_list]
        score_rougeL = np.mean(scores_rougeL)
        print(f"ROUGE: {score_rouge1:.6f}")
    if args.use_cosine:
        scores_cosine = [score["cosine"] for score in scores_list]
        score_cosine = np.mean(scores_cosine)
        print(f"Cosine: {score_cosine:.6f}")
    if args.use_comet:
        scores_comet = [score["comet"] for score in scores_list]
        score_comet = np.mean(scores_comet)
        print(f"COMET: {score_comet:.6f}")
    else:
        score_comet = 0

    scores_llm = [score["llm"] for score in scores_list]
    score_llm = np.mean(scores_llm)
    print(f"LLM: {score_llm:.6f}")

    head_score = [{"bleu": score_bleu, "rouge": score_rouge1, "cosine": score_cosine, "comet": score_comet, "llm": score_llm}]
    
    if args.use_bleu:
        scores_bleu = [score["bleu"] for score in base_scores_list]
        score_bleu = np.mean(scores_bleu)
        print(f"BLEU: {score_bleu:.6f}")
    if args.use_rouge:
        scores_rouge1 = [score["rouge1"] for score in base_scores_list]
        score_rouge1 = np.mean(scores_rouge1)
        scores_rougeL = [score["rougeL"] for score in base_scores_list]
        score_rougeL = np.mean(scores_rougeL)
        print(f"ROUGE: {score_rouge1:.6f}")
    if args.use_cosine:
        scores_cosine = [score["cosine"] for score in base_scores_list]
        score_cosine = np.mean(scores_cosine)
        print(f"Cosine: {score_cosine:.6f}")
    if args.use_comet:
        scores_comet = [score["comet"] for score in base_scores_list]
        score_comet = np.mean(scores_comet)
        print(f"COMET: {score_comet:.6f}")
    else:
        score_comet = 0

    scores_llm = [score["llm"] for score in base_scores_list]
    score_llm = np.mean(scores_llm)
    print(f"LLM: {score_llm:.6f}")

    base_head_score = [{"bleu": score_bleu, "rouge": score_rouge1, "cosine": score_cosine, "comet": score_comet, "llm": score_llm}]
    # acc_head[f"topk"] = head_score
    # acc_head[f"acc"] = acc_num / len(selected_prompts)
    #print(acc_head)
    print(f"done")
    #mode = os.path.basename(path).split('_')[1]
    data = [args.function_name[0], args.model_name, args.use_head, args.mask_ratio, head_score, acc_num / len(prompts), args.token_use, args.alpha, args.sign]
    base_data= [args.function_name[0], args.model_name, args.use_head, args.mask_ratio, base_head_score, acc_num / len(prompts), args.token_use, args.alpha, args.sign]

    # with open(f"abla_results/{args.function_name}/head_acc_{args.function_name}.json", "w") as f:
    #     json.dump(data, f)
    with open(f"./{args.folder_dir}/{args.model_name}/{args.output_dir}", "a") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(data) 
        writer.writerow(base_data) 
    print(f"write done")