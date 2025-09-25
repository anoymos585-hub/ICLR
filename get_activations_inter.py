import argparse
from tqdm import tqdm
import json
import torch
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText
import openai
# Specific pyvene imports
from utils import tokenized_tqa, cot_prompt, get_qwen_activations_pyvene, get_intern_activations_pyvene, get_gemma_activations_pyvene, cot_prompt_inter
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration, Gemma3nForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from transformers.image_utils import load_image

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
    'gemma-3n-e2b-it': 'google/gemma-3n-e2b-it',
    'gemma-3n-e4b-it': 'google/gemma-3n-e4b-it'
}      

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gemma-3n-e2b-it')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='extra_300')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--use_setoken', default=False, action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    if "Qwen" in args.model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name_or_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    elif "Intern" in args.model_name:
        tokenizer = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    elif "gemma" in args.model_name:
        print("load gemma")
        tokenizer = AutoProcessor.from_pretrained(model_name_or_path)
        model = Gemma3nForConditionalGeneration.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float32, device_map="cuda:0", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    device = "cuda:0"

    dataset1 = json.load(open(f'./dataset/train_1000_final.json'))
    dataset2 = json.load(open(f'./dataset/test_1000_final.json'))
    dataset = dataset1 + dataset2

    prompts, answers, images = cot_prompt_inter(dataset, args.dataset_name)
    # print(prompts)
    # print(labels)
    # print(answers)
    collectors = []
    pv_config = []
    
    if "Intern" in args.model_name:
        print(model)
        for layer in range(model.config.text_config.num_hidden_layers): 
            collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
            collectors.append(collector)
            pv_config.append({
                "component": f"model.language_model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(collector),
            })
    elif "gemma" in args.model_name:
        print(model)
        #print(model.language_model)
        for layer in range(model.config.text_config.num_hidden_layers): 
            collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
            collectors.append(collector)
            pv_config.append({
                "component": f"model.language_model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(collector),
            })
    else:
        # print(model)
        # print(model.language_model)
        # print(model.language_model.model)
        # print(model.model.language_model.layers)
        for layer in range(model.config.num_hidden_layers): 
            collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
            collectors.append(collector)
            pv_config.append({
                "component": f"model.language_model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(collector),
            })
    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = [] 
    output_answers = []
    output_llm = []
    token_positions_list = []
    print("Getting activations")
    i = 0
    save_labels = []
    for i, prompt in tqdm(enumerate(prompts)):
        # prompt = tokenizer(prompt, return_tensors = 'pt').input_ids
        if "Qwen" in args.model_name:
            head_wise_activations, _, output_answer, token_positions = get_qwen_activations_pyvene(tokenizer, collected_model, collectors, prompt, device, args, images[i])
        elif "Intern" in args.model_name:
            head_wise_activations, _, output_answer, token_positions = get_intern_activations_pyvene(tokenizer, collected_model, collectors, prompt, device, args, images[i])
        elif "gemma" in args.model_name:
            head_wise_activations, _, output_answer, token_positions = get_gemma_activations_pyvene(tokenizer, collected_model, collectors, prompt, device, args, images[i])
        # all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        # all_layer_wise_activations.append(layer_wise_activations.copy())
        all_head_wise_activations.append(head_wise_activations.copy())
        output_answers.append(output_answer)
        output_llm.append([{"prompt": prompt, "truth": answers[i], "answer": output_answer, "image_path": images[i]}])
        #save_labels.append(labels[i])
        token_positions_list.append(token_positions)
        i = i + 1

    with open("./inter_results/output_" + args.model_name + "_" + args.dataset_name + "_" + args.mode + ".json", "w") as f:
        json.dump(output_llm, f, indent=4)
        
    print("Saving labels")
    #np.save(f'{args.model_name}_{args.dataset_name}_labels_{args.mode}.npy', save_labels)
    # np.save(f'{args.model_name}_{args.dataset_name}_token_positions_{args.mode}.npy', token_positions_list)
    
    with open(f'./inter_results/{args.model_name}_{args.dataset_name}_token_positions_{args.mode}.pkl', 'wb') as f:
        pickle.dump(token_positions_list, f)

    # print("Saving layer wise activations")
    # np.save(f'../features/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    quarter = len(prompts) // 4
    # np.save(f'{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations, allow_pickle=True)
    if ("7B" in args.model_name or "8B" in args.model_name or "4b" in args.model_name) and args.mode=="train":
        with open(f'./inter_results/{args.model_name}_{args.dataset_name}_head_wise_{args.mode}_0.pkl', 'wb') as f:
            pickle.dump(all_head_wise_activations[0:quarter], f)
        with open(f'./inter_results/{args.model_name}_{args.dataset_name}_head_wise_{args.mode}_1.pkl', 'wb') as f:
            pickle.dump(all_head_wise_activations[quarter:2*quarter], f)
        with open(f'./inter_results/{args.model_name}_{args.dataset_name}_head_wise_{args.mode}_2.pkl', 'wb') as f:
            pickle.dump(all_head_wise_activations[2*quarter:3*quarter], f)
        with open(f'./inter_results/{args.model_name}_{args.dataset_name}_head_wise_{args.mode}_3.pkl', 'wb') as f:
            pickle.dump(all_head_wise_activations[3*quarter:], f)
    else:
        with open(f'./inter_results/{args.model_name}_{args.dataset_name}_head_wise_{args.mode}.pkl', 'wb') as f:
            pickle.dump(all_head_wise_activations, f)


if __name__ == '__main__':
    main()
