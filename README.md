# DO VISION-LANGUAGE MODELS REASON LIKE HUMANS? EXPLORING THE FUNCTIONAL ROLES OF ATTENTION HEADS
Repository for CogVision


## Setup

### Files and Package Required
Run ```pip install -r requirements.txt```

Generate an openai api key, and insert the key to the file "utils.py" at the line:
client = OpenAI(api_key="your key here")
Also insert the key to "negative_intervention/utils.py" and "positive_intervention/utils.py" 

Run ```./start_vllm.sh``` to start the vllm backend server, this llm is used to selecting topk tokens from answers and to determine the answer correctness as the llm-judge

### First Step: Get Activations
Please run the following command **sequentially**

Run ```python get_activations.py --mode train```
Run ```python get_activations.py --mode test```


Run ```python get_topk_tokens_by_answer.py --mode train```
Run ```python get_topk_tokens_by_answer.py --mode test```

Run ```python logistic_regression.py```

### Second Step: Negative Intervention
Run ```cd negative_intervention```
Run ```./ablation.sh``` for negative intervention

Run ```./ablation_other_function.sh``` for negative intervention of other functions topk

Run ```./ablation_OOD.sh``` for negative intervention of OOD datasets

Run ```./ablation_cot.sh``` for negative intervention of influence of low-level cognitive heads for high-order function


### Third Step: Positive Intervention
Run ```python get_activations_inter.py```
Run ```python get_topk_tokens_by_answer_inter.py```

Run ```cd positive_intervention```
Run ```./ablation_task_recorrect_ID.sh``` for positive intervention in domain

Run ```./ablation_task_recorrect_OOD.sh``` for positive intervention out of domain


