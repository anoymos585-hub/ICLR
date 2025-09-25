import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Gemma3nForConditionalGeneration, AutoModelForImageTextToText
from PIL import Image
import numpy as np
import json
from qwen_vl_utils import process_vision_info
import argparse
import random
from openai import OpenAI
from tqdm import tqdm
client = OpenAI(
    api_key='EMPTY',
    base_url=f'http://0.0.0.0:30000/v1',
)
gpt_model = client.models.list().data[0].id
print(f'model: {gpt_model}')

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
    'gemma-3n-e2b-it': 'google/gemma-3n-e2b-it',
    'gemma-3n-e4b-it': 'google/gemma-3n-e4b-it'
}    

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

Only output your answer:
"""

gpt_prompt = """
You are given a question, reference answer and a predicted answer to a sub-question.

This is the question to answer:
<question>
{question}
</question>

This is the predicted answer:
<prediction_answer>
{prediction_answer}
</prediction_answer>


Please identify at most 5 important words ONLY from the predicted answer, if less that 5 words in predicted answer, just return any number of words that are important in the predicted answer. Return only the words, separated by commas, without any explanation.
"""

def cot_prompt(dataset, low_function_name=None, high_function_name=None):
    all_prompts = []
    all_labels = []
    all_answers = []
    all_images = []
    all_subquestions = []
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
            all_subquestions.append(subquestion)
            all_prompts.append(prompt_text)
            all_labels.append(label)
            all_answers.append(answer)
            all_images.append(image)
    return all_prompts, all_labels, all_answers, all_images, all_subquestions


def extract_attention_scores(model, processor, question, subquestion, image_path, target_layer_heads, head, cognitive_skill, model_name):
    """
    提取Qwen2.5-VL模型第一个输出token对所有输入的attention score
    """
    
    # 加载图像
    print(f"Loading image from {image_path}...")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": subquestion},
            ],
        }
    ]
    
    # 处理输入
    if "Qwen" in model_name:
        start_image_token = "<|vision_start|>"
        end_image_token = "<|vision_end|>"
        actual_token_idx = 14
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    elif "Intern" in model_name:
        start_image_token = "<img>"
        end_image_token = "</img>"
        actual_token_idx = 3
        inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
    elif "gemma" in model_name:
        start_image_token = "<start_of_image>"
        end_image_token = "<end_of_image>"
        actual_token_idx = 4
        inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

    inputs = inputs.to("cuda")
    
    # 获取输入token信息
    input_ids = inputs["input_ids"][0]
    
    # 解码输入tokens用于显示
    tokenizer = processor.tokenizer
    input_tokens = []
    
    print("\n=== Input Analysis ===")
    print(f"Total input length: {len(input_ids)}")
    if len(input_ids) > 3000:
        return None
    # 识别图像token的位置
    image_token_id = tokenizer.convert_tokens_to_ids(start_image_token)
    vision_end_token_id = tokenizer.convert_tokens_to_ids(end_image_token)
    #print(tokenizer.convert_ids_to_tokens(input_ids[0:14]))
    image_start_idx = None
    image_end_idx = None
    
    for i, token_id in enumerate(input_ids):
        if token_id == image_token_id:
            image_start_idx = i
        elif token_id == vision_end_token_id:
            image_end_idx = i
            break
    
    print(f"Image tokens range: {image_start_idx} to {image_end_idx}")
    
    # 生成一个token并获取attention
    #print("\n=== Generating complete response ===")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True
        )
    
    # 解码生成的文本
    generated_ids = outputs.sequences[0]
    input_length = len(input_ids)
    generated_tokens = generated_ids[input_length:]  # 只取新生成的部分
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #print(f"Generated text: {generated_text}")
    prompt_text = gpt_prompt.format(
                question=subquestion,
                prediction_answer=generated_text
            )
    messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt_text}
        ]}]

    resp = client.chat.completions.create(model=gpt_model, messages=messages, max_tokens=512, temperature=0)
    #print(resp)
    response = resp.choices[0].message.content
    topk_words = [word.strip() for word in response.split(',')]
    #print(f"Top-k important words: {topk_words}")

    def find_word_token_positions(words, generated_tokens, tokenizer):
        """
        找到重要词汇在生成的tokens中的位置
        """
        word_positions = []
        generated_text_tokens = [tokenizer.decode([token]) for token in generated_tokens]
        
        for word in words:
            # 编码单个词汇
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            word_text_tokens = [tokenizer.decode([token]) for token in word_tokens]
            
            # 在生成的tokens中寻找匹配
            positions = []
            for i in range(len(generated_text_tokens) - len(word_text_tokens) + 1):
                # 检查是否匹配 (考虑部分匹配和大小写)
                match = True
                for j, word_token in enumerate(word_text_tokens):
                    if word_token.lower().strip() not in generated_text_tokens[i + j].lower().strip():
                        match = False
                        break
                
                if match:
                    # 添加对应的绝对位置 (相对于完整序列)
                    for j in range(len(word_text_tokens)):
                        positions.append(input_length + i + j)
            
            if positions:
                word_positions.extend(positions)
                #print(f"Word '{word}' found at positions: {[pos - input_length for pos in positions]} (relative), {positions} (absolute)")
            else:
                print(f"Warning: Word '{word}' not found in generated tokens")
        
        return list(set(word_positions))  # 去重

    topk_positions = find_word_token_positions(topk_words, generated_tokens, tokenizer)
    #print(f"All topk word positions: {topk_positions}")
    if not topk_positions:
        print("Warning: No important words found in generated tokens. Using last few tokens as fallback.")
        # 如果没找到，使用最后几个token作为备选
        topk_positions = list(range(len(generated_ids) - min(5, len(generated_tokens)), len(generated_ids)))

    results = []
    attention_matrix = []
    attention_matrix_scores = []
    # 获取所有生成步骤的attention
    all_attentions = outputs.attentions  # List of tuples, each tuple contains attentions for one generation step
    
    for layer_idx, head_idx in target_layer_heads:
        #print(f"\n--- LAYER {layer_idx}, HEAD {head_idx} ---")
        
        # 收集该层该头在重要token位置的所有attention
        layer_head_attentions = []
        
        for step_idx, step_attentions in enumerate(all_attentions):
            # 检查当前步骤生成的token是否在重要位置中
            current_token_pos = input_length + step_idx
            
            if current_token_pos in topk_positions:
                # 检查layer和head索引是否有效
                if layer_idx >= len(step_attentions):
                    print(f"ERROR: Layer {layer_idx} does not exist (max: {len(step_attentions)-1})")
                    continue
                
                layer_attention = step_attentions[layer_idx]
                num_heads = layer_attention.shape[1]
                
                if head_idx >= num_heads:
                    print(f"ERROR: Head {head_idx} does not exist (max: {num_heads-1})")
                    continue
                
                # 获取当前生成token对输入序列的attention
                # attention_weights: [batch, heads, seq_len, seq_len]
                attention_weights = layer_attention[0, head_idx]  # [seq_len, seq_len]
                
                # 获取当前生成token（最后一行）对输入序列的attention
                current_attention = attention_weights[-1, :input_length]  # 只关注对输入的attention
                layer_head_attentions.append(current_attention.to(torch.float32).cpu().numpy())
                
                #print(f"Step {step_idx}, token pos {current_token_pos}: attention shape {current_attention.shape}")
        
        if layer_head_attentions:
            # 计算重要tokens的attention平均值
            avg_attention = np.mean(layer_head_attentions, axis=0)
            
            # 标准化 (从位置14开始，跳过特殊token)
            arr = avg_attention[actual_token_idx:] if len(avg_attention) > actual_token_idx else avg_attention
            #arr = (arr - arr.min()) / (arr.max() - arr.min())
            visual_attention = float(arr[:image_end_idx-actual_token_idx].sum() / arr.sum())
            attention_matrix.append(arr)
            attention_matrix_scores.append(visual_attention)
            #print(f"Collected {len(layer_head_attentions)} attention vectors from important tokens")
            #print(f"Averaged attention shape: {arr.shape}")
        else:
            print(f"No important tokens found for layer {layer_idx}, head {head_idx}")
            # 添加零向量或跳过
            if len(attention_matrix) > 0:
                attention_matrix.append(np.zeros_like(attention_matrix[0]))
            else:
                print("Skipping this layer-head combination")

    return {
        'input_tokens': input_ids,
        'image_token_range': (image_start_idx, image_end_idx),
        'tokenizer': tokenizer,
        'image_end_idx': image_end_idx,
        "attention_matrix_scores": attention_matrix_scores
    }

# 使用示例
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--head', type=str, default="topk")
    parser.add_argument('--pos', type=str, default="topk")
    parser.add_argument('--head_num', type=int, default=30)
    #parser.add_argument('--function', type=str, default="Inference")
    args = parser.parse_args()
    functions_names = ["Vision Knowledge Recall", "Language Knowledge Recall", "Semantic Understanding", "Math Reasoning", "Low-Level Vision Reception", "Inference", "High-Level Vision Reception", "Decision-Making"]
    end_indices = {}
    random.seed(42)
    model_name = HF_NAMES[args.model_name]
    if "Qwen" in args.model_name:
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype="auto", device_map="auto", trust_remote_code=True, output_attentions=True)
    elif "Intern" in args.model_name:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True, output_attentions=True)
    elif "gemma" in args.model_name:
        print("load gemma")
        processor = AutoProcessor.from_pretrained(model_name)
        model = Gemma3nForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype="auto", device_map="cuda:0", trust_remote_code=True, output_attentions=True)

    with open(f'./model_config.json', "r") as f:
        model_config = json.load(f)
    for config in model_config:
        if config["model_name"] == args.model_name:
            layer_num = config["layer_num"]
            heads_num = config["heads_num"]
            dim = config["dim"]
            break
    dataset = json.load(open(f'.dataset/test_1000_final.json'))
    prompts, labels, answers, images, all_subquestions = cot_prompt(dataset)

    for function in functions_names:
        
        #model_name = HF_NAMES[args.model_name]
        attention_matrix_scores = []
        
        
        with open(f"./main_results/{args.model_name}/importance_{args.model_name}_topk_indices_True.json", "r") as f:
            importance_matrix = json.load(f)
        finished = False
        #prompts, labels, answers, images, all_subquestions = cot_prompt(dataset)
        #print(len(prompts))
        for i, prompt in tqdm(enumerate(prompts)):
            torch.cuda.empty_cache()
            question = prompt
            image_path = images[i]
            cognitive_skill = labels[i]
            if cognitive_skill == function:
                # print("question: ", question)
                # print("cognitive_skill: ", cognitive_skill)
                if args.head == "topk":
                    selected = importance_matrix[cognitive_skill][0:args.head_num]
                elif args.head == "randomk":
                    selected = random.sample(importance_matrix[cognitive_skill][args.head_num:], args.head_num)
                target_layer_heads = []
                for each_head in selected:
                    first = each_head // heads_num
                    second = each_head % heads_num
                    target_layer_heads.append((first, second))
                with torch.no_grad():
                    result = extract_attention_scores(
                        model=model,
                        processor=processor,
                        question=question,
                        subquestion=all_subquestions[i],
                        image_path=image_path,
                        target_layer_heads=target_layer_heads,
                        head=args.head,
                        cognitive_skill=cognitive_skill,
                        model_name=args.model_name
                    )
                if result == None:
                    continue
                attention_matrix_scores.append(result["attention_matrix_scores"])
        print("\n=== Analysis Complete ===")
        #end_indices[function] = result["image_end_idx"]
        average_scores = [sum(col) / len(col) for col in zip(*attention_matrix_scores)]
        with open(f"./main_results/{args.model_name}/attention_matrix_{function}.json", "w") as f:
            json.dump(average_scores, f, indent=4)
    # with open(f"./main_results/{args.model_name}/image_end_indices.json", "w") as f:
    #     json.dump(end_indices, f, indent=4)