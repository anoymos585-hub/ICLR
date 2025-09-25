# Utils to work with pyvene

import os
import sys
sys.path.insert(0, "TruthfulQA")
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
# import llama
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
# import llama
import pandas as pd
import warnings
# from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
# from baukit import Trace, TraceDict
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
from functools import partial
import random
from torch.cuda.amp import autocast
import time

import openai
import re
import json
import evaluate
# from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from openai import OpenAI
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

torch._dynamo.config.suppress_errors = True
# Or completely disable dynamo
torch._dynamo.config.disable = True

client = OpenAI(api_key="your key here")

ENGINE_MAP = {
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'circulus/alpaca-7b',
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf',
    'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
}



def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"

def format_truthfulqa_end_q(question, choice, rand_question): 
    return f"Q: {question} A: {choice} Q: {rand_question}"


def tokenized_tqa(dataset, tokenizer): 

    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i]['mc2_targets']['choices']
        labels = dataset[i]['mc2_targets']['labels']

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)): 
            choice = choices[j]
            label = labels[j]
            prompt = format_truthfulqa(question, choice)
            if i == 0 and j == 0: 
                print(prompt)
            prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
            all_prompts.append(prompt)
            all_labels.append(label)
    
    return all_prompts, all_labels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_intern(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# def split_model(model_name):
#     device_map = {}
#     world_size = torch.cuda.device_count()
#     config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#     num_layers = config.llm_config.num_hidden_layers
#     # Since the first GPU will be used for ViT, treat it as half a GPU.
#     num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
#     num_layers_per_gpu = [num_layers_per_gpu] * world_size
#     num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
#     layer_cnt = 0
#     for i, num_layer in enumerate(num_layers_per_gpu):
#         for j in range(num_layer):
#             device_map[f'language_model.model.layers.{layer_cnt}'] = i
#             layer_cnt += 1
#     device_map['vision_model'] = 0
#     device_map['mlp1'] = 0
#     device_map['language_model.model.tok_embeddings'] = 0
#     device_map['language_model.model.embed_tokens'] = 0
#     device_map['language_model.output'] = 0
#     device_map['language_model.model.norm'] = 0
#     device_map['language_model.model.rotary_emb'] = 0
#     device_map['language_model.lm_head'] = 0
#     device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

#     return device_map

def get_llama_activations_bau(model, prompt, device): 
    HEADS = [f"model.layers.{i}.self_attn.head_out" for i in range(model.config.num_hidden_layers)]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS+MLPS) as ret:
        # with TraceDict(model, HEADS+MLPS, retain_input=True) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()
        mlp_wise_hidden_states = [ret[mlp].output.squeeze().detach().cpu() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def find_subsequence(subseq_o, seq):
    index = -1
    """Return the start index of subseq in seq, or -1 if not found"""
    if len(subseq_o) > 1:
        subseq = subseq_o[:-1]  # 去掉首尾的 [ 和 ]
    else:
        subseq = subseq_o
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i:i+len(subseq)] == subseq:
            index = i
            return i
    if index == -1:
        subseq = subseq_o[1:] 
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i:i+len(subseq)] == subseq:
                index = i -1
                return i-1
    if index == -1:
        subseq = subseq_o[1:-1] 
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i:i+len(subseq)] == subseq:
                index = i -1
                return i-1
    return -1

class StopOnCloseBracket(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_token_id = tokenizer.convert_tokens_to_ids("]")

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id

def safe_json_parse(json_text: str) -> dict:
    # 检查 "answer": 后面的值是否有双引号
    def fix_answer_value(match):
        key, value = match.group(1), match.group(2).strip()
        # 如果已经是以引号开头和结尾，就不加引号
        if value.startswith('"') and value.endswith('"'):
            return f'{key}{value}'
        else:
            # 否则添加双引号
            return f'{key}"{value}"'

    # 匹配 "answer": 后跟非引号开头的值
    fixed_text = re.sub(r'("answer": )(.+?)(?=\s*})', fix_answer_value, json_text)

    return json.loads(fixed_text)

def get_qwen_activations_pyvene(tokenizer, collected_model, collectors, prompt, device, args, image_path):
    output_answers = []
    no_output = []
    stopping_criteria = StoppingCriteriaList([StopOnCloseBracket(tokenizer.tokenizer)])
    with torch.no_grad():
        messages = [
            {"role": "user", "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ]}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        generated_ids, adv_generated = collected_model.generate(
            base=model_inputs,
            max_new_tokens=128,
            do_sample=False,
            output_original_output=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.tokenizer.eos_token_id,

        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = tokenizer.tokenizer.decode(output_ids[0:], skip_special_tokens=True).strip("\n")
        #print("all content: ", content)
        output_answer = None
        matches = re.findall(r'\[\s*{[^}]+}\s*\]', content)
        if matches:
            output_json_block = matches[-1].strip()
            #print("Extracted JSON answer block:", output_json_block)

            try:
                parsed = safe_json_parse(output_json_block)
                # parsed = json.loads(output_json_block)
                # output_answer = parsed[0]["answer"]  # This will return '"$80"'
                if isinstance(parsed, list) and isinstance(parsed[0], dict) and "answer" in parsed[0]:
                    output_answer = parsed[0]["answer"]
                else:
                    output_answer = None 
                #print("Final extracted answer (with quotes):", output_answer)
            except json.JSONDecodeError:
                output_answer = None
                print("Failed to decode JSON.")
        else:
            print("No JSON block found.")

    # 获取生成部分的 token 位置信息
    # 获取完整生成的 token ids
    generated_ids = generated_ids[0]  # [seq_len]

    # 对 output_text 编码为 token ids，不包含 special tokens
    # 去除 prompt 部分
    generated_only_ids = generated_ids[model_inputs["input_ids"].shape[1]:].tolist()
    if output_answer is None:
        output_answer = tokenizer.tokenizer.decode(generated_only_ids, skip_special_tokens=True)
            
    try:
        # 确保是字符串
        output_answer_str = str(output_answer).strip()
        output_text_ids = tokenizer.tokenizer.encode(output_answer_str, add_special_tokens=False)
    except Exception as e:
        print(f"Encoding failed for output_answer={output_answer}. Error: {e}")
        output_text_ids = []
    # output_text_ids = tokenizer.encode(output_answer, add_special_tokens=False)


    # 查找位置
    relative_start = find_subsequence(output_text_ids, generated_only_ids)
    if relative_start != -1:
        # absolute_start = prompt["input_ids"].shape[1] + relative_start
        absolute_start = relative_start
        token_positions = list(range(absolute_start, absolute_start + len(output_text_ids)))
        #print("Token positions for output_text:", token_positions)
    else:
        print("output_text tokens not found in generated sequence.")
        token_positions = list(range(len(generated_only_ids)-1))
     

    # 提取 attention head 的 activations
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            # collector.states: [num_tokens, num_heads, dim]
            #print(collector.states)
            states_per_gen = torch.stack(collector.states, axis=0)  # shape: [T, H, D]
            if args.use_setoken:
                selected_states = states_per_gen[token_positions]      # shape: [5, H, D]
            else:
                selected_states = states_per_gen
            
            #print(selected_states.shape)
            head_wise_hidden_states.append(selected_states.cpu().numpy())
        else:
            print("Got None Activations")
            head_wise_hidden_states.append(None)
        collector.reset()

    mlp_wise_hidden_states = []  # 如果你有 MLP collector，也可以加在这里
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).numpy()
    #print(head_wise_hidden_states.shape)
    return head_wise_hidden_states, mlp_wise_hidden_states, output_answer, token_positions

def get_intern_activations_pyvene(tokenizer, collected_model, collectors, prompt, device, args, image_path):
    output_answers = []
    no_output = []
    #stopping_criteria = StoppingCriteriaList([StopOnCloseBracket(tokenizer._tokenizer)])
    with torch.no_grad():
        messages = [
            {"role": "user", "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ]}
        ]

        model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device, dtype=torch.float16)
        #print(model_inputs)
        generated_ids, adv_generated = collected_model.generate(
            base=model_inputs,
            max_new_tokens=128,
            do_sample=False,
            output_original_output=True,
            pad_token_id=tokenizer.tokenizer.eos_token_id,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = tokenizer.tokenizer.decode(output_ids[0:], skip_special_tokens=True).strip("\n")
        print("all content: ", content)
        output_answer = None
        matches = re.findall(r'\[\s*{[^}]+}\s*\]', content)
        if matches:
            output_json_block = matches[-1].strip()
            print("Extracted JSON answer block:", output_json_block)

            try:
                # output_json_block = re.sub(r'("answer": )(\$[0-9]+)', r'\1"\2"', output_json_block)
                # output_json_block = re.sub(r'("answer": )([^\"]\S.*?)(?=\s*})',lambda m: f'{m.group(1)}"{m.group(2).strip()}"', output_json_block)
                parsed = safe_json_parse(output_json_block)
                # parsed = json.loads(output_json_block)
                # output_answer = parsed[0]["answer"]  # This will return '"$80"'
                if isinstance(parsed, list) and isinstance(parsed[0], dict) and "answer" in parsed[0]:
                    output_answer = parsed[0]["answer"]
                else:
                    output_answer = None 
                print("Final extracted answer (with quotes):", output_answer)
            except json.JSONDecodeError:
                output_answer = None
                print("Failed to decode JSON.")
        else:
            print("No JSON block found.")


        print("Generated output:", content)


    # 获取生成部分的 token 位置信息
    # 获取完整生成的 token ids
    generated_ids = generated_ids[0]  # [seq_len]

    # 对 output_text 编码为 token ids，不包含 special tokens
    # 去除 prompt 部分
    generated_only_ids = generated_ids[model_inputs["input_ids"].shape[1]:].tolist()
    if output_answer is None:
        output_answer = tokenizer.tokenizer.decode(generated_only_ids, skip_special_tokens=True)
            
    try:
        # 确保是字符串
        output_answer_str = str(output_answer).strip()
        output_text_ids = tokenizer.tokenizer.encode(output_answer_str, add_special_tokens=False)
    except Exception as e:
        print(f"Encoding failed for output_answer={output_answer}. Error: {e}")
        output_text_ids = []
    # output_text_ids = tokenizer.encode(output_answer, add_special_tokens=False)


    # 查找位置
    relative_start = find_subsequence(output_text_ids, generated_only_ids)
    if relative_start != -1:
        # absolute_start = prompt["input_ids"].shape[1] + relative_start
        absolute_start = relative_start
        token_positions = list(range(absolute_start, absolute_start + len(output_text_ids)))
        print("Token positions for output_text:", token_positions)
    else:
        print("output_text tokens not found in generated sequence.")
        token_positions = list(range(len(generated_only_ids)-1))


    # 提取 attention head 的 activations
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            # collector.states: [num_tokens, num_heads, dim]
            #print(collector.states)
            states_per_gen = torch.stack(collector.states, axis=0)  # shape: [T, H, D]
            if args.use_setoken:
                selected_states = states_per_gen[token_positions]      # shape: [5, H, D]
            else:
                selected_states = states_per_gen
            
            #print(selected_states.shape)
            head_wise_hidden_states.append(selected_states.cpu().numpy())
        else:
            print("Got None Activations")
            head_wise_hidden_states.append(None)
        collector.reset()

    mlp_wise_hidden_states = []  # 如果你有 MLP collector，也可以加在这里
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).numpy()
    print(head_wise_hidden_states.shape)
    return head_wise_hidden_states, mlp_wise_hidden_states, output_answer, token_positions

def get_gemma_activations_pyvene(tokenizer, collected_model, collectors, prompt, device, args, image_path):
    output_answers = []
    no_output = []
    stopping_criteria = StoppingCriteriaList([StopOnCloseBracket(tokenizer.tokenizer)])
    with torch.no_grad():
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]

        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        generated_ids, adv_generated = collected_model.generate(
            base=model_inputs,
            max_new_tokens=128,
            do_sample=False,
            output_original_output=True
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = tokenizer.tokenizer.decode(output_ids[0:], skip_special_tokens=True).strip("\n")
        print("content: ", content)
        output_answer = None
        matches = re.findall(r'\[\s*{[^}]+}\s*\]', content)
        if matches:
            output_json_block = matches[-1].strip()
            print("Extracted JSON answer block:", output_json_block)

            try:
                # output_json_block = re.sub(r'("answer": )(\$[0-9]+)', r'\1"\2"', output_json_block)
                # output_json_block = re.sub(r'("answer": )([^\"]\S.*?)(?=\s*})',lambda m: f'{m.group(1)}"{m.group(2).strip()}"', output_json_block)
                parsed = safe_json_parse(output_json_block)
                # parsed = json.loads(output_json_block)
                # output_answer = parsed[0]["answer"]  # This will return '"$80"'
                if isinstance(parsed, list) and isinstance(parsed[0], dict) and "answer" in parsed[0]:
                    output_answer = parsed[0]["answer"]
                else:
                    output_answer = None 
                print("Final extracted answer (with quotes):", output_answer)
            except json.JSONDecodeError:
                output_answer = None
                print("Failed to decode JSON.")
        else:
            print("No JSON block found.")

        print("Generated output:", content)


    # 获取生成部分的 token 位置信息
    # 获取完整生成的 token ids
    generated_ids = generated_ids[0]  # [seq_len]

    # 对 output_text 编码为 token ids，不包含 special tokens
    # 去除 prompt 部分
    generated_only_ids = generated_ids[model_inputs["input_ids"].shape[1]:].tolist()
    if output_answer is None:
        output_answer = tokenizer.tokenizer.decode(generated_only_ids, skip_special_tokens=True)
        
    try:
        # 确保是字符串
        output_answer_str = str(output_answer).strip()
        output_text_ids = tokenizer.tokenizer.encode(output_answer_str, add_special_tokens=False)
    except Exception as e:
        print(f"Encoding failed for output_answer={output_answer}. Error: {e}")
        output_text_ids = []
    # output_text_ids = tokenizer.encode(output_answer, add_special_tokens=False)


    # 查找位置
    relative_start = find_subsequence(output_text_ids, generated_only_ids)
    if relative_start != -1:
        # absolute_start = prompt["input_ids"].shape[1] + relative_start
        absolute_start = relative_start
        token_positions = list(range(absolute_start, absolute_start + len(output_text_ids)))
        print("Token positions for output_text:", token_positions)
    else:
        print("output_text tokens not found in generated sequence.")
        token_positions = list(range(len(generated_only_ids)-1))
    
    # 判断output是否是正确的
     

    # 提取 attention head 的 activations
    #print(collectors)
    head_wise_hidden_states = []
    for collector in collectors:
        #print(collector.collect_state)
        if collector.collect_state:
            # collector.states: [num_tokens, num_heads, dim]
            #print(collector.states)
            states_per_gen = torch.stack(collector.states, axis=0)  # shape: [T, H, D]
            if args.use_setoken:
                selected_states = states_per_gen[token_positions]      # shape: [5, H, D]
            else:
                selected_states = states_per_gen
            
            #print(selected_states.shape)
            head_wise_hidden_states.append(selected_states.cpu().numpy())
        else:
            print("Got None Activations")
            head_wise_hidden_states.append(None)
        collector.reset()

    mlp_wise_hidden_states = []  # 如果你有 MLP collector，也可以加在这里
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).numpy()
    #print(head_wise_hidden_states.shape)
    return head_wise_hidden_states, mlp_wise_hidden_states, output_answer, token_positions


def get_llama_logits(model, prompt, device): 

    model.eval()
    with torch.no_grad(): 
        prompt = prompt.to(device)
        logits = model(prompt).logits
        logits = logits.detach().cpu()
        return logits

system_role = """
You are an expert in analytical and logical reasoning. Your task is to answer the question.
"""

prompt = """
{system_role}
Here is the question which is a subquestion of original question:
<question>
{question}
</question>

Here is the prior knowledge in chain-of-thought (CoT) format:
<original_question>
{original_question}
</original_question>

<prior_knowledge>
{cot}
</prior_knowledge>

Instructions:
- Answer the question carefully.
- You can use the information in the original question and prior_knowledge to help you answer the question.
- Your response should be clear and concise.
- Do not include any explanation, commentary, or code.
- Do not output anything after the closing square bracket `]`.

Only output your final answer using this format:
[
    {{"answer": "<Your answer here>"}}
]

Your answer:
"""

prompt_inter = """
{system_role}
Here is the question:
<question>
{question}
</question>

Instructions:
- Answer the question carefully.
- Your response should be clear and concise.
- Do not include any explanation, commentary, or code.
- Do not output anything after the closing square bracket `]`.

Only output your final answer using this format:
[
    {{"answer": "<Your answer here>"}}
]

Your answer:
"""
def cot_prompt(dataset, low_function_name=None, high_function_name=None):
    all_prompts = []
    all_labels = []
    all_answers = []
    all_images = []

    # define logic type mapping
    infor_extract = ["Retrieval", "Knowledge Recall", "Semantic Understanding", "Syntactic Understanding"]
    higher_logic = ["Induction", "Inference", "Logical Reasoning", "Decision-making"]

    for i in range(len(dataset)):
        question = dataset[i]['question']
        generated = dataset[i]['generated']
        image = dataset[i]['image_path']
        #image = ""
        #print(generated)
        for j in range(len(generated)):
            subquestion = generated[j]['subquestion']
            cot = ""

            # Accumulate all previous sub-QA pairs for context
            if j > 0:                
                for k in range(j):
                    if generated[k]["cognitive_skill"] != "Retrieval":
                        continue
                    prev_subq = generated[k]['subquestion']
                    prev_ans = generated[k]['answer']
                    cot += f"Q{k+1}: {prev_subq}\nA{k+1}: {prev_ans}\n"
            else:
                cot += "No prior knowledge.\n"

            # label: 0 for lower-level (information extraction), 1 for higher-level reasoning
            cognitive_skill = generated[j]['cognitive_skill']
            # label = 0 if cognitive_skill in infor_extract else 1
            label = cognitive_skill
            
            answer = generated[j]['answer']
            # format prompt
            prompt_text = prompt.format(
                system_role=system_role,
                original_question=question,
                question=subquestion,
                cot=cot.strip()
            )

            all_prompts.append(prompt_text)
            all_labels.append(label)
            all_answers.append(answer)
            all_images.append(image)
    return all_prompts, all_labels, all_answers, all_images

def cot_prompt_inter(dataset, dataset_name, low_function_name=None, high_function_name=None):
    all_prompts = []
    all_labels = []
    all_answers = []
    all_images = []

    # define logic type mapping
    infor_extract = ["Retrieval", "Knowledge Recall", "Semantic Understanding", "Syntactic Understanding"]
    higher_logic = ["Induction", "Inference", "Logical Reasoning", "Decision-making"]
    
    for i in range(len(dataset)):
        if (dataset_name == "okvqa" and "okvqa" in dataset[i]['image_path']) or (dataset_name == "Math_Vision" and "Math-Vision" in dataset[i]['image_path']) or (dataset_name == "Math_Vista" and "Math-Vision" in dataset[i]['image_path']) or (dataset_name == "visulogic" and "visulogic" in dataset[i]['image_path']):
            question = dataset[i]['question'].replace("<image>", "")
            generated = dataset[i]['generated']
            image = dataset[i]['image_path']

                
            answer = dataset[i]['final_answer']
            # format prompt
            prompt_text = prompt_inter.format(
                system_role=system_role,
                question=question
            )

            all_prompts.append(prompt_text)
            all_answers.append(answer)
            all_images.append(image)
    return all_prompts, all_answers, all_images


def cot_prompt_OOD(dataset, check_dataset, low_function_name=None, high_function_name=None):
    all_prompts = []
    all_labels = []
    all_answers = []
    all_images = []
    all_questions = []
    all_subquestions = []
    # define logic type mapping
    infor_extract = ["Retrieval", "Knowledge Recall", "Semantic Understanding", "Syntactic Understanding"]
    higher_logic = ["Induction", "Inference", "Logical Reasoning", "Decision-making"]
    idx = 0
    for i in range(len(dataset)):
        prompt_text = dataset[i]['messages'][0]["content"].replace("<image>","")
        answer = dataset[i]['messages'][1]["content"]
        #generated = dataset[i]['generated']
        image = dataset[i]['images'][0]
        if image in check_dataset:
            continue
        

        all_prompts.append(prompt_text)

        all_answers.append(answer)
        all_images.append(image)
        idx += 1
        if idx == 100:
            break
    return all_prompts, all_answers, all_images

def get_filtered_data(dataset, model, tokenizer):
    prompts, labels = cot_prompt(dataset)
    responses = []
    for i, input_text in enumerate(prompts):
        # Prepare input data and move to GPU
        # inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024).cuda()
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {key: tensor.cuda() for key, tensor in inputs.items()}  # Move each tensor to CUDA

        # Generate text
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.0, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )

        # Decode generated text
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Print the generated text
        print("Generated Text:\n", decoded)
        answer_start = decoded.find("Your answer:")
        if answer_start != -1:
            response_text = decoded[answer_start + len("Your answer:"):].strip()
        else:
            response_text = decoded.strip()

        responses.append(response_text)    
    # Filter out responses that are not in the expected format
    filtered_responses = []
    
    return filtered_responses, labels  

def adv_generate(collected_model, tokenizer, prompt, device, image, model_name):
    stopping_criteria = StoppingCriteriaList([StopOnCloseBracket(tokenizer)])
    with torch.no_grad():
        if "Qwen" in model_name:
            messages = [
                {"role": "user", "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ]}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            model_inputs = tokenizer(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
        elif "Intern" in model_name:
            messages = [
                {"role": "user", "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ]}
            ]

            model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device, dtype=torch.float16)

        elif "gemma" in model_name:
            image = load_image(image)
            model_inputs = tokenizer(text="<image>"+prompt, images=image, return_tensors="pt").to(device, dtype=torch.float32)
        base_generated, adv_generated = collected_model.generate(
            base=model_inputs,
            max_new_tokens=128,
            do_sample=False,
            output_original_output=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

    return base_generated, adv_generated

def get_answer(base_generated, tokenizer): 
     
    output_text = tokenizer.decode(base_generated[0], skip_special_tokens=True)
    matches = re.findall(r'\[\s*{[^}]+}\s*\]', output_text)
    output_answer = None
    if matches:
        output_json_block = matches[-1].strip()
        # print("Extracted JSON answer block:", output_json_block)

        try:
            # output_json_block = re.sub(r'("answer": )(\$[0-9]+)', r'\1"\2"', output_json_block)
            # output_json_block = re.sub(r'("answer": )([^\"]\S.*?)(?=\s*})',lambda m: f'{m.group(1)}"{m.group(2).strip()}"', output_json_block)
            parsed = safe_json_parse(output_json_block)
            if isinstance(parsed, list) and isinstance(parsed[0], dict) and "answer" in parsed[0]:
                output_answer = parsed[0]["answer"]
            else:
                output_answer = None 
            # parsed = json.loads(output_json_block)
            # output_answer = parsed[0]["answer"]  # This will return '"$80"'
            print("Final extracted answer:", output_answer)
        except:
            print("Failed to decode JSON.")
    else:
        print("No JSON block found.")
        
    return output_answer

# def kl_divergence(base_pd, mask_pd):
#     epsilon = 1e-12
#     base_pd = base_pd + epsilon
#     mask_pd = mask_pd + epsilon

#     now_kl_divergence = torch.sum(base_pd * torch.log(base_pd / mask_pd))

#     return now_kl_divergence

def kl_divergence(base_pd, mask_pd):
    epsilon = 1e-8

    # 避免零概率，确保数值稳定
    base_pd = torch.clamp(base_pd, min=epsilon)
    mask_pd = torch.clamp(mask_pd, min=epsilon)

    # 计算KL散度
    now_kl_divergence = torch.sum(base_pd * torch.log(base_pd / mask_pd))

    return now_kl_divergence

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge", experiment_id=f"comet_{os.getpid()}_{int(time.time()*1000)}")
comet = evaluate.load("comet", experiment_id=f"comet_{os.getpid()}_{int(time.time()*1000)}")

def emb_similarity(texts):
    # 使用 OpenAI API 获取文本的嵌入向量
    texts = [each[0] for each in texts]
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )

    # 提取嵌入向量
    embeddings = [np.array(e.embedding) for e in response.data]

    # 计算余弦相似度
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarity = cosine_similarity(embeddings[0], embeddings[1])
    return similarity

def evaluate_metrics_intervention(predictions, references, predictions_full, references_full, sources=None, args=None, evaluate_model=None):
    """
    统一评估函数：支持 BLEU、ROUGE、COMET
    参数:
        predictions: List[str]，模型输出
        references: List[str] 或 List[List[str]]，参考答案（BLEU 支持多参考）
        sources: List[str]，源文本（仅 COMET 需要）
    返回:
        dict，包含 bleu, rouge1, rouge2, rougeL, comet（如有）
    """
    predictions = [p if p.strip() != "" else "N/A" for p in predictions]
    references = [r if r.strip() != "" else "N/A" for r in references]
    results = {}

    # BLEU（支持多个参考）

    bleu_score = bleu.compute(predictions=predictions, references=sources)
    results["bleu"] = bleu_score["bleu"]

    # ROUGE
    rouge_score = rouge.compute(predictions=predictions, references=sources)
    results.update(rouge_score)  # rouge1, rouge2, rougeL, rougeLsum
    
    # 计算cosine similarity
    # emb_prediction = evaluate_model.encode(predictions, convert_to_tensor=True)
    # emb_reference = evaluate_model.encode(sources, convert_to_tensor=True)
    cosine_score = emb_similarity([predictions, sources])
    results["cosine"] = cosine_score

    # COMET（需要源句）
    if args.use_comet and sources is not None:
        comet_score = comet.compute(
            predictions=predictions,
            references=references,
            sources=sources
        )
        results["comet"] = comet_score["mean_score"]
    else:
        results["comet"] = "Skipped (no sources provided)"

    return results

def evaluate_metrics(predictions, references, predictions_full, references_full, sources=None, args=None, evaluate_model=None):
    """
    统一评估函数：支持 BLEU、ROUGE、COMET
    参数:
        predictions: List[str]，模型输出
        references: List[str] 或 List[List[str]]，参考答案（BLEU 支持多参考）
        sources: List[str]，源文本（仅 COMET 需要）
    返回:
        dict，包含 bleu, rouge1, rouge2, rougeL, comet（如有）
    """
    predictions = [p if p.strip() != "" else "N/A" for p in predictions]
    references = [r if r.strip() != "" else "N/A" for r in references]
    results = {}

    # BLEU（支持多个参考）

    bleu_score = bleu.compute(predictions=predictions_full, references=references_full)
    results["bleu"] = bleu_score["bleu"]

    # ROUGE
    rouge_score = rouge.compute(predictions=predictions, references=sources)
    results.update(rouge_score)  # rouge1, rouge2, rougeL, rougeLsum
    
    # 计算cosine similarity
    # emb_prediction = evaluate_model.encode(predictions, convert_to_tensor=True)
    # emb_reference = evaluate_model.encode(sources, convert_to_tensor=True)
    cosine_score = emb_similarity([predictions, sources])
    results["cosine"] = cosine_score

    # COMET（需要源句）
    if args.use_comet and sources is not None:
        comet_score = comet.compute(
            predictions=predictions,
            references=references,
            sources=sources
        )
        results["comet"] = comet_score["mean_score"]
    else:
        results["comet"] = "Skipped (no sources provided)"

    return results

def simi_scoring(prompt, label, source_answer, image, model_name, tokenizer, intervened_model, device, args, evaluate_model):
    prompt = tokenizer(prompt, return_tensors='pt')    
    prompt = prompt.to(device)
    # output = collected_model({"input_ids": prompt.input_ids, "output_hidden_states": True})[1]
    prompt = {k: v.cuda() for k, v in prompt.items()}

    base_generated, adv_generated = adv_generate(intervened_model, tokenizer, prompt, device, image, model_name)
    base_tokens = base_generated.sequences
    adv_tokens = adv_generated.sequences
    # base_logits = base_generated.scores
    # adv_logits = adv_generated.scores
        
    base_answer = get_answer(base_tokens, tokenizer)
    adv_answer = get_answer(adv_tokens, tokenizer)
    # base_answer, adv_answer = None, None
    #print(base_answer)
    #print(adv_answer)
    if args.use_bleu:
        base_generated_only_ids = base_tokens[0][prompt["input_ids"].shape[1]:].tolist()
        # adv_generated_ids = adv_tokens[0]
        adv_generated_only_ids = adv_tokens[0][prompt["input_ids"].shape[1]:].tolist()
        base_answer_full = tokenizer.decode(base_generated_only_ids, skip_special_tokens=True)
    # if adv_answer is None:
        adv_answer_full = tokenizer.decode(adv_generated_only_ids, skip_special_tokens=True)   
    if base_answer is None or adv_answer is None:
        base_answer, adv_answer = base_answer_full, adv_answer_full


    scores = evaluate_metrics([str(adv_answer)], [str(base_answer)], [str(adv_answer_full)], [str(base_answer_full)], sources=[str(source_answer)], args=args, evaluate_model=evaluate_model)  

    if "kl_scores" not in locals():
        kl_scores = []
    return base_answer, adv_answer, scores, kl_scores


def simi_scoring_intervention(prompt, label, source_answer, image, model_name, tokenizer, intervened_model, device, args, evaluate_model):
    prompt = tokenizer(prompt, return_tensors='pt')    
    prompt = prompt.to(device)
    # output = collected_model({"input_ids": prompt.input_ids, "output_hidden_states": True})[1]
    prompt = {k: v.cuda() for k, v in prompt.items()}

    base_generated, adv_generated = adv_generate(intervened_model, tokenizer, prompt, device, image, model_name)
    base_tokens = base_generated.sequences
    adv_tokens = adv_generated.sequences
    # base_logits = base_generated.scores
    # adv_logits = adv_generated.scores
        
    base_answer = get_answer(base_tokens, tokenizer)
    adv_answer = get_answer(adv_tokens, tokenizer)
    # base_answer, adv_answer = None, None
    #print(base_answer)
    #print(adv_answer)
    if args.use_bleu:
        base_generated_only_ids = base_tokens[0][prompt["input_ids"].shape[1]:].tolist()
        # adv_generated_ids = adv_tokens[0]
        adv_generated_only_ids = adv_tokens[0][prompt["input_ids"].shape[1]:].tolist()
        base_answer_full = tokenizer.decode(base_generated_only_ids, skip_special_tokens=True)
    # if adv_answer is None:
        adv_answer_full = tokenizer.decode(adv_generated_only_ids, skip_special_tokens=True)   
    if base_answer is None or adv_answer is None:
        base_answer, adv_answer = base_answer_full, adv_answer_full

    scores = evaluate_metrics_intervention([str(adv_answer)], [str(base_answer)], [str(adv_answer_full)], [str(base_answer_full)], sources=[str(source_answer)], args=args, evaluate_model=evaluate_model)  

    if "kl_scores" not in locals():
        kl_scores = []

    return base_answer, adv_answer, scores, kl_scores


def get_intervation_result(prompt, tokenizer, intervened_model, device, args):
    prompt = tokenizer(prompt, return_tensors='pt')    
    prompt = prompt.to(device)
    # output = collected_model({"input_ids": prompt.input_ids, "output_hidden_states": True})[1]
    prompt = {k: v.cuda() for k, v in prompt.items()}

    # base_generated, adv_generated = adv_generate(intervened_model, tokenizer, prompt, device)
    stopping_criteria = StoppingCriteriaList([StopOnCloseBracket(tokenizer)])
    with torch.no_grad():
        base_generated, adv_generated = intervened_model.generate(
            base=prompt,
            max_new_tokens=1024,
            do_sample=False,
            output_original_output=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id
        )

    base_tokens = base_generated
    adv_tokens = adv_generated

    base_generated_only_ids = base_tokens[0][prompt["input_ids"].shape[1]:].tolist()
    adv_generated_only_ids = adv_tokens[0][prompt["input_ids"].shape[1]:].tolist()
    base_answer_full = tokenizer.decode(base_generated_only_ids, skip_special_tokens=True)
    adv_answer_full = tokenizer.decode(adv_generated_only_ids, skip_special_tokens=True)   

    
    print("Base answer:", base_answer_full, "Adv answer:", adv_answer_full)
    return base_answer_full, adv_answer_full
    # return base_answer_full, adv_answer_full, [base_thinking_content, base_content], [adv_thinking_content, adv_content]



def kl_evaluate(logits_base, logits_adv, base_start_ids, adv_start_ids):
    
    # 计算 KL 散度 of the first token
    pd_base = torch.softmax(logits_base[0][-1, :], dim=-1)       
    pd_adv = torch.softmax(logits_adv[0][-1, :], dim=-1)

    score_0 = kl_divergence(pd_base, pd_adv)
    
    # 计算 KL 散度 of the first answer token
    pd_base = torch.softmax(logits_base[base_start_ids][-1, :], dim=-1)
    pd_adv = torch.softmax(logits_adv[adv_start_ids][-1, :], dim=-1)
    score_k = kl_divergence(pd_base, pd_adv)
    
    # 计算 KL 散度 of all tokens
    pd_base = torch.softmax(logits_base, dim=-1)
    pd_adv = torch.softmax(logits_adv, dim=-1)
    
    score = kl_divergence(torch.sum(pd_base, dim=0), torch.sum(pd_adv, dim=0))
    
    print(f"KL divergence: {score_0.item():.8f}")

    
    return [score_0.item(), score_k.item(), score.item()]

def kl_scoring(prompt, label, model, tokenizer, intervened_model, device):
    prompt = tokenizer(prompt, return_tensors='pt')    
    prompt = prompt.to(device)

    with torch.no_grad():
        with autocast(dtype=torch.bfloat16):
            logits_base = model(input_ids=prompt["input_ids"], attention_mask=prompt["attention_mask"]).logits
            logits = intervened_model({"input_ids": prompt.input_ids, "output_hidden_states": True})[1].logits
        pd_base = torch.softmax(logits_base[:, -1, :], dim=-1)       
        pd_adv = torch.softmax(logits[:, -1, :], dim=-1)

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(input_ids=generated, attention_mask=attention_mask)
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

            next_token_logits = logits[:, -1, :]  # 拿最后一个token的logits（刚生成的）
            
            # 可以保存logits
            if step == 5:  # 举例：拿第6个生成的新token
                logits_6th_token = next_token_logits

            # 选下一个token（比如greedy选最大logits）
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 拼接到generated里
            generated = torch.cat([generated, next_token], dim=-1)

            # attention_mask 也要更新
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
                dim=1
            )

    score = kl_divergence(pd_base, pd_adv)
    
    print(f"KL divergence: {score.item():.8f}")

    
    return score

def get_separated_activations(labels, head_wise_activations): 

    # separate activations by question
    dataset=load_dataset('truthful_qa', 'multiple_choice')['validation']
    actual_labels = []
    for i in range(len(dataset)):
        actual_labels.append(dataset[i]['mc2_targets']['labels'])

    idxs_to_split_at = np.cumsum([len(x) for x in actual_labels])        

    labels = list(labels)
    separated_labels = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
    assert separated_labels == actual_labels

    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)

    return separated_head_wise_activations, separated_labels, idxs_to_split_at

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
            usable_head_wise_activations = np.concatenate([head_wise_activations[i].reshape(num_layers, num_heads, -1)[layer,head,:] for i in usable_idxs], axis=0)
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions