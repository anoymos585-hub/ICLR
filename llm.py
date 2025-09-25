
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import openai
import json
import os.path as osp
import asyncio
#from asyncapi import process_api_requests_from_file
import torch
import logging
import ast
import matplotlib.pyplot as plt
import copy

import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata
import utils
import pandas as pd
import ast

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import re
import random
import numpy as np

system_role = """
You are an expert in analytical and logical reasoning. You will be given a main question and an image along with its full answer, 
and a subquestion and answer.
You task is to evaluate the correctness of the answer of the subquestion based on the main question, image and CoT.
"""
@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    return request_url.split("/")[-1]


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

def append_to_json(_dict,path):
    with open(path, 'ab+') as f:
        f.seek(0,2)                                #Go to the end of file
        if f.tell() == 0 :                         #Check if file is empty
            f.write(json.dumps([_dict], default=str, indent=4).encode())  #If empty, write an array
        else :
            f.seek(-1,2)
            f.truncate()                           #Remove the last character, open the array
            f.write(' , '.encode())                #Write the separator
            f.write(json.dumps(_dict, default=str, indent=4).encode())    #Dump the dictionary
            f.write(']'.encode())




def num_tokens_consumed_from_request(request_json, model="gpt-3.5-turbo"):
    if type(request_json) is dict:
        messages = request_json['messages']
    else:
        messages = request_json
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    if model == "gpt-3.5-turbo":
        tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    result = []

    async def call_API(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        if "aug_id" in self.request_json:
            aug_id = copy.deepcopy(self.request_json['aug_id'])
            del self.request_json['aug_id']
        else:
            aug_id = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                # self.result['aug_id'] = aug_id
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                # append_to_json(self.result, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            response['aug_id'] = aug_id
            new_response = {}
            for key,value in response.items():
                if type(value) is str:
                    new_response[key] = value.replace("\n", "\\n")
                else:
                    new_response[key] = value
            append_to_json(new_response, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")

async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
    budget:int
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # intialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    data = pd.read_csv(requests_filepath).to_dict(orient='records')[:budget]
    data = [{**i, "messages":ast.literal_eval(i['messages'])} for i in data]

    # `requests` will provide requests one at a time
    requests = data.__iter__()
    logging.debug(f"File opened. Entering main loop")

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if queue_of_requests_to_retry.empty() is False:
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
            elif file_not_finished:
                try:
                    # get new request
                    # print(next(requests))
                    request_json = next(requests)
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(request_json),
                        attempts_left=max_attempts,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Read file exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_API(
                        request_url=request_url,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        save_filepath=save_filepath,
                        status_tracker=status_tracker,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

    # after finishing, log final status
    logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")

async def call_async_api(request_filepath, save_filepath, request_url, api_key, max_request_per_minute, max_tokens_per_minute):
    await process_api_requests_from_file(
            requests_filepath=request_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_request_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name='cl100k_base',
            max_attempts=int(2),
            logging_level=int(logging.INFO),
            budget=-1
        )


def generate_chat_input_file(input_text, system_role, model_name = 'gpt-3.5-turbo', temperature = 0, n = 1):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['model'] = model_name
        obj['messages'] = [
            {"role": "system", "content": system_role},
            {
                'role': 'user',
                'content': text 
            }
        ]
        obj['temperature'] = temperature
        obj['n'] = n
        jobs.append(obj)
    return jobs 

def openai_text_api(input_text, api_key, model_name = "gpt-3.5-turbo", temperature = 0, n = 1):
    response = openai.ChatCompletion.create(
        model=model_name,               
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": input_text},
        ],
        temperature=temperature,
        api_key=api_key,
        n = n)
    return response 


def efficient_openai_text_api(input_text, input_file, output_file, system_role, api_key="change_this_to_your_key", model_name="gpt-3.5-turbo", request_url = "https://api.openai.com/v1/chat/completions", rewrite = True, temperature = 0, n = 1):
    # import ipdb; ipdb.set_trace()
    # openai_result = []
    rewrite = True
    non_empty_results = []
    results = []
    if not osp.exists(output_file):
        jobs = generate_chat_input_file(input_text, system_role, model_name=model_name, temperature = temperature, n = n)

        with open(input_file, "w") as f:
            for i, job in enumerate(jobs):
                json_string = json.dumps(job)
                if job['messages'][0]['content'] != "":
                    f.write(json_string + "\n")
                    non_empty_results.append(i)
        asyncio.run(
            call_async_api(
                input_file, save_filepath=output_file,
                request_url=request_url,
                api_key=api_key,
                max_request_per_minute=60, 
                max_tokens_per_minute=10000
            )
        )
    openai_result = []

    with open(output_file, 'r') as f:
        # import ipdb; ipdb.set_trace()
        for i, line in enumerate(f):
            json_obj = json.loads(line.strip())
            content = json_obj[1]
            idx = json_obj[-1]
            choices = []
            non_empty_results.append(i)
            if content == "":
                openai_result.append(("", idx))
                # import ipdb; ipdb.set_trace()
            elif isinstance(idx, int):
                choices = [x['message']['content'] for x in json_obj[1]['choices']]
                openai_result.append((choices, idx))
                # input = json_obj[0]['messages'][0]['content']
                # input = input.split('\n')[1]
                # texts_input.append(input)
            else:
                idx = json_obj[-2]
                new_result = openai_text_api(json_obj[0]['messages'][0]['content'], api_key, model_name = json_obj[0]['model'], temperature = json_obj[0]['temperature'], n = json_obj[0]['n'])
                choices = [x['message']['content'] for x in new_result['choices']]
                openai_result.append((choices, idx))
                # input = json_obj[0]['messages'][0]['content']
                # input = input.split('\n')[1]
                # texts_input.append(input)
    openai_result = sorted(openai_result, key=lambda x:x[-1])
    results = [("", idx) for idx in range(len(input_text))]
    # for i, r in enumerate(openai_result):
    #     results[non_empty_results[i]] = r
    return openai_result