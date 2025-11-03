import sys
import os
BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)

from openai import OpenAI
import json
import itertools
import time
import tqdm
from src.utils import new_directory, load_jsonl
import argparse

OPENAI_API_BASE="YOUR_OPENAI_API_BASE"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
ENGINE_ID="YOUR_ENGINE_ID"


def api_request(messages, engine, client, backend, **kwargs):
    """
    Calls the underlying LLM endpoint based on the backend.
    If an error is encountered (e.g. error 429), sleep for 20 seconds before retrying.
    """
    while True:
        completion = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0),
                    max_tokens=kwargs.get("max_tokens", 512),
                    top_p=kwargs.get("top_p", 1),
                    frequency_penalty=kwargs.get("frequency_penalty", 0),
                    presence_penalty=kwargs.get("presence_penalty", 0),
                    stop=kwargs.get("stop", None),
                )
        token_usage = {
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
        return None, completion.choices[0].message.content, token_usage

def api_infer_single(
        prompt, 
        max_token_length=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        return_format="",):
    messages = [{"role": "user", "content": prompt}]
    client = OpenAI(
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
    )
    backend = "openai"
    kwargs = {
        "temperature": temperature,
        "max_tokens": max_token_length,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stop": stop,
        "return_format": return_format,
    }
    return api_request(messages, ENGINE_ID, client, backend, **kwargs)
    
def api_infer(
        prompt_jsonl, 
        max_token_length=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        return_format="",):
    
    prompts = [content['prompt'] for content in prompt_jsonl]
    print(prompts[0])
    
    data_list = []
    for idx in tqdm.tqdm(range(len(prompts))):
        prompt = prompts[idx]

        reasoning_content, content, token_usage = api_infer_single(
            prompt = prompt,
            max_token_length=max_token_length,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            return_format=return_format
        )
        info = {}
        info['idx'] = idx
        info["prompt"] = prompt
        info["response"] = content
        info["reasoning_content"] = reasoning_content if reasoning_content else ""
        info["token_usage"] = token_usage if token_usage else ""
        data_list.append(info)
    return data_list
        
def write_response(data, output_path):
    """
    Append a single response (as JSON) to the output file.
    """
    directory_path = os.path.dirname(output_path)
    new_directory(directory_path)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        



