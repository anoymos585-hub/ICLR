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
# from lib.SHIPS.pd_diff import kl_divergence
# from lib.SHIPS.ships_utils import sort_ships_dict
# from lib.SHIPS.get_ships import SHIPS
from collections import defaultdict
from tqdm import tqdm
import math
import numpy as np
import random
from utils import cot_prompt, kl_scoring, simi_scoring, cot_prompt_only_correct
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from interveners import wrapper, Collector, ITI_Intervener, head_Intervener
import pyvene as pv
from sentence_transformers import SentenceTransformer
import csv
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from openai import OpenAI

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# 关闭 huggingface tokenizers 的 parallelism 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PL_NO_SLURM"] = "1"
os.environ["SLURM_JOB_ID"] = ""
os.environ["SLURM_ARRAY_TASK_ID"] = ""
# 提高 Tensor Core matmul 精度，加速推理/训练
torch.set_float32_matmul_precision('high')

print("[环境设置] TOKENIZERS_PARALLELISM=false, matmul_precision=high 已设置")

gpt_client = OpenAI(
    api_key='EMPTY',
    base_url=f'http://0.0.0.0:30000/v1',
)
gpt_model = gpt_client.models.list().data[0].id
print(f'model: {gpt_model}')

gpt_prompt = """
Return True if the prediction answer is semantically consistent or similar to the ground truth answer. If the prediction does not match, or if it contains gibberish, unreadable words, or placeholder text such as <Your answer here>, return False.

This is the ground truth answer: {reference_answer}
This is the predicted answer: {prediction_answer}

Respond ONLY with True or False.
"""

HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
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

system_role = """
You are an expert in analytical visual reasoning. You will be given an image and some prior knowledge in chain-of-thought (CoT) format.
Your task is to answer the question using the information provided.
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
- Use the information in the original question and prior_knowledge to answer the question carefully.
- Your response should be clear and concise.
- Do not include any explanation, commentary, or code.
- Do not output anything after the closing square bracket `]`.

Only output your final answer using this format:
[
    {{"answer": "<Your answer here>"}}
]

Your answer:
"""

# # 定义干预函数
def run_intervention(inter_heads_idx):
#     # 初始化 head_mask 全为 0 (干预)
#     head_mask = torch.ones((num_layers, num_heads), dtype=torch.float32).to(model.device)
    
#     # 设置指定 head 的干预 (这里用 1 表示只用该 head)
#     head_mask[layer_idx, 0:10] = 0.0
    
#     # 创建干预器
#     intervener = head_Intervener(head_mask=head_mask)    
# #     intervener = ITI_Intervener(direction, args.alpha) #head=-1 to collect all head activations, multiplier doens't matter
# #     interveners.append(intervener)
#     pv_config = [{
#         "component": f"model.layers[{layer_idx}].self_attn.o_proj.input",
#         "intervention": wrapper(intervener),
#     }]
    pv_config = []
    for layer_idx, heads in inter_heads_idx.items():
        # 初始化 head_mask 全为 1 (不干预)
        head_mask = torch.ones(head_dim * num_heads, dtype=torch.float32).to(model.device)
        # 设置指定 head 的干预
        for head in heads:
            head_mask[head * head_dim:(head + 1) * head_dim] = 0.0001
        intervener = head_Intervener(head_mask=head_mask)   
        intervener.layer_idx = layer_idx  # 注意：为了让 __call__ 能访问当前层索引 
        pv_config.append({
            "component": f"model.language_model.layers[{layer_idx}].self_attn.o_proj.input",
            "intervention": wrapper(intervener),
        })

    intervened_model = pv.IntervenableModel(pv_config, model)
    
    return intervened_model

# def run_intervention(model, layer_idx, head_idx, num_layers, num_heads):
#     # 创建 head_mask，默认 1，指定位置置为 0
#     head_mask = torch.ones((num_layers, num_heads), dtype=torch.float32).to(model.device)
#     head_mask[layer_idx, head_idx] = 0.0

#     # 创建干预器并传入当前层索引
#     intervener = head_Intervener(head_mask)
#     intervener.layer_idx = layer_idx  # 注意：为了让 __call__ 能访问当前层索引

#     # 注册干预器到指定层
#     pv_config = [{
#         "component": f"model.layers[{layer_idx}].self_attn.o_proj.input",
#         "intervention": wrapper(intervener),
#     }]
#     pv.apply_config(model, pv_config)  # 注意这里是注册到 model，不返回新模型

#     return model

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

def find_elbow_point(y):
    # x = np.arange(len(y))
    # y = np.array(y)

    # # 拟合首尾直线
    # line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    # line_vec = line_vec / np.linalg.norm(line_vec)
    
    # # 向量化点到首点的连线
    # vecs = np.stack([x - x[0], y - y[0]], axis=1)
    # proj_lengths = np.dot(vecs, line_vec)
    # proj_points = np.outer(proj_lengths, line_vec)
    # dists = np.linalg.norm(vecs - proj_points, axis=1)

    # elbow_index = np.argmax(dists)
    count = 0
    for each in y:
        if each > 0.85:
            count += 1 
    elbow_index = count
    return elbow_index

functions_names = ["Vision Knowledge Recall", "Language Knowledge Recall", "Semantic Understanding", "Math Reasoning", "Low-Level Vision Reception", "Inference", "High-Level Vision Reception", "Decision-Making"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--dataset_name', type=str, default='cot_qa')
    parser.add_argument('--token_use', type=str, default="topk")
    parser.add_argument('--mask_ratio', type=int, default=58)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--function_name', type=str, default='Math Reasoning')
    parser.add_argument('--use_bleu', default=True, action='store_true')
    parser.add_argument('--use_rouge', default=True, action='store_true')
    parser.add_argument('--use_cosine', default=True, action='store_true')
    parser.add_argument('--use_comet', default=True, action='store_true')
    parser.add_argument('--use_kl', default=False, action='store_true')
    parser.add_argument('--use_head', type=str, default="topk")
    parser.add_argument('--use_layer_bias', default=False, action="store_true")
    parser.add_argument('--output_dir', type=str, default="head_acc.csv")
    parser.add_argument('--folder_dir', type=str, default="cot_results")
    parser.add_argument('--low_function_name', type=str, default="High-Level Vision Reception")
    parser.add_argument('--high_function_name', type=str, default="Math Reasoning")

    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)

    model_name_or_path = HF_NAMES[args.model_name]
    if "Qwen" in args.model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name_or_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    elif "Intern" in args.model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    elif "paligemma2" in args.model_name:
        print("load paligemma2")
        tokenizer = PaliGemmaProcessor.from_pretrained(model_name_or_path)
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    device = "cuda"
    # define number of layers and heads
    # num_layers = model.config.num_hidden_layers
    # num_heads = model.config.num_attention_heads
    # hidden_size = model.config.hidden_size
    # if args.model_name == "qwen3-4B":
    #     head_dim = model.config.head_dim
    # else:
    #     head_dim = hidden_size // num_heads    
    with open(f'../model_config.json', "r") as f:
        model_config = json.load(f)
    for config in model_config:
        if config["model_name"] == args.model_name:
            num_layers = config["layer_num"]
            num_heads = config["heads_num"]
            head_dim = config["dim"]
            break
    args.mask_ratio = math.ceil(0.1 * num_layers * num_heads)
    if "e4b" in args.model_name:
        args.mask_ratio = args.mask_ratio * 2
    if "InternVL3-2B" in args.model_name:
        args.mask_ratio = math.ceil(args.mask_ratio / 2)
    mask_config = {
        "scale_factor": 0.0001,
        "mask_type": "scale_mask",
    }
    evaluate_model = SentenceTransformer('all-MiniLM-L6-v2')

    dataset2 = json.load(open(f'../dataset/test_1000_final.json'))
    dataset = dataset2
    dataset_name = f'test_1000'
    check_dataset = json.load(open(f'../head_results/output_{args.model_name}_test_1000_test_with_gpt_label.json'))
    function_idx = []  
    global_idx = 0
    for i in range(len(dataset)):
        if str(check_dataset[global_idx][0]["label"]).lower() == "false":
            global_idx += 1
            continue
        generated_list = dataset[i]["generated"]  
        labels = []      
        for j in range(len(generated_list)): 
            labels.append(generated_list[j]["cognitive_skill"])
        #print()
        if args.high_function_name in labels and args.low_function_name in labels and labels.index(args.low_function_name) < labels.index(args.high_function_name):
            function_idx.append(i)
        global_idx += 1

    data = [dataset[i] for i in function_idx]
    # prompts, labels, answers, images, original_questions, subquestions = cot_prompt_only_correct(dataset, check_dataset)

    with open(f"../main_results/{args.model_name}/importance_{args.model_name}_{args.token_use}_indices_{args.use_layer_bias}.json", "r") as f:
        importances = json.load(f)

    with open(f"../main_results/{args.model_name}/importance_{args.model_name}_{args.token_use}_scores_{args.use_layer_bias}.json", "r") as f:
        importance_scores = json.load(f)
    elbow = {}
    sorted_indices = {}
    for label in functions_names:
        importance_score = importance_scores[label]
        sorted_indices[label] = sorted(enumerate(importance_score), key=lambda x: abs(x[1]), reverse=True)
        elbow[label] = find_elbow_point([abs(i[1]) for i in sorted_indices[label]])

    print("elbow number: ", elbow)
    head_features = []
    if args.use_head == "randomk":
        #head_features = [i[0] for i in sorted_indices[args.function_name][0:elbow[args.function_name]]]
        head_features = importances[args.function_name][:args.mask_ratio]
        all_heads = list(range(num_layers * num_heads))
        all_heads = [h for h in all_heads if h not in head_features]
        head_features = random.sample(all_heads, len(head_features))
    elif args.use_head == "lowk":
        head_features = importances[args.function_name][-args.mask_ratio:]
    elif args.use_head == "topk":
        head_features = importances[args.function_name][:args.mask_ratio]
    elif args.use_head == "elbow":
        head_features = [i[0] for i in sorted_indices[args.function_name][0:elbow[args.function_name]]]

        # selected = [my_list[i] for i in importances[args.function_name]]
        # head_features = importances[args.function_name][:args.mask_ratio]
    else:
        excluded_head_indices = [h[0] for h in importances[args.function_name][:args.mask_ratio]]
        remaining_heads = [h for h in importances[args.use_head] if h[0] not in excluded_head_indices]
        head_features = remaining_heads[:args.mask_ratio]
        # head_features = importances[args.use_head][:args.mask_ratio]
        # head_features = importances[args.function_name][args.mask_ratio:2*args.mask_ratio]
        
    # 遍历每层每个 head 进行干预    acc_head = {}
    # for i in tqdm(range(num_layers), desc="Layers"):
    #     for j in range(num_heads):
    #         inter_heads_idx = get_intervention_heads(i, j, num_layers, num_heads)
    if args.use_head != "last":
        inter_heads_idx = get_topk_intervention_heads(head_features)  # 这里假设你只想干预第 0 层的第 0 个头
    elif args.use_head == "last":
        inter_heads_idx = get_last_intervention_heads(head_features)
    
    intervened_model = run_intervention(inter_heads_idx)

    scores_list = []
    scores_list_normal = []
    generated_answers = []
    acc_head = {}
    acc_num = 0
    acc_num_normal = 0
    
    if not os.path.exists(os.path.join(f"{args.folder_dir}", args.high_function_name.replace(" ", "_"))):
        os.makedirs(os.path.join(f"{args.folder_dir}", args.high_function_name.replace(" ", "_")))

    for i in tqdm(range(len(data))):
        question = data[i]['question']
        generated = data[i]['generated']
        image = "../"+data[i]['image_path']
        adv_answer_list = []
        for j in range(len(generated)):
            subquestion = generated[j]['subquestion']
            cot = ""
            cot_normal = ""
            # Accumulate all previous sub-QA pairs for context
            if j > 0:                
                for k in range(j):
                    prev_subq = generated[k]['subquestion']
                    prev_ans = adv_answer_list[k]
                    #prev_ans_normal = generated[k]['answer']
                    cot += f"Q{k+1}: {prev_subq}\nA{k+1}: {prev_ans}\n"
                    #cot_normal += f"Q{k+1}: {prev_subq}\nA{k+1}: {prev_ans_normal}\n"
            else:
                cot += "No prior knowledge.\n"
                #cot_normal += "No prior knowledge.\n"

            # label: 0 for lower-level (information extraction), 1 for higher-level reasoning
            cognitive_skill = generated[j]['cognitive_skill']
            # label = 0 if cognitive_skill in infor_extract else 1
            label = cognitive_skill            
            answer = generated[j]['answer']
            prompt_text = prompt.format(
                system_role=system_role,
                original_question=question,
                cot=cot.strip(),
                question=subquestion
            )

            base_answer, adv_answer, scores, kl_scores = simi_scoring(prompt_text, label, answer, image, args.model_name, tokenizer, intervened_model, device, args, evaluate_model)


            adv_answer_list.append(adv_answer)
            if label in [args.high_function_name]:
                llm_judge_prompt = gpt_prompt.format(
                        reference_answer=answer,
                        prediction_answer=base_answer,
                    )
                #print(llm_judge_prompt)
                messages = [{'role': 'user', 'content': [
                    {'type': 'text', 'text': llm_judge_prompt}
                ]}]

                resp = gpt_client.chat.completions.create(model=gpt_model, messages=messages, max_tokens=512, temperature=0)
                response = resp.choices[0].message.content
                print("LLM response: ", response)
                if "true" in response.lower():
                    scores["llm"] = 1.0
                else:
                    scores["llm"] = 0.0
                scores_list.append(scores)

                if scores["bleu"] > 0.8 or scores["rouge1"] > 0.6 or scores["cosine"] > 0.6:
                    acc_num += 1
                # generated_answers.append([base_answer, adv_answer])
                with open(os.path.join(args.folder_dir, args.high_function_name.replace(" ", "_"), args.low_function_name.replace(" ", "_") + "_" + args.use_head + "_head_answer.txt"), "a") as f:
                    f.write(f"{args.use_head}: {adv_answer} || {base_answer} || {answer}\n")
                with open(os.path.join(args.folder_dir, args.high_function_name.replace(" ", "_"), args.low_function_name.replace(" ", "_") + "_" + args.use_head +"_head_score.txt"), "a") as f:
                    f.write(f"{args.use_head}: {scores}\n")

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
    acc_head[f"topk"] = head_score
    acc_head[f"acc"] = acc_num / len(scores_list)
    print(acc_head)
    print(f"done")
    #mode = os.path.basename(path).split('_')[1]
    data_written = [args.low_function_name, args.high_function_name, args.model_name, args.use_head, args.mask_ratio, acc_head["topk"], acc_head["acc"], args.token_use, "adv"]

    with open(f"{args.folder_dir}/{args.output_dir}", "a") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(data_written) 
    print(f"write done")
