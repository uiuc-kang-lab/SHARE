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

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ENGINE_ID = os.environ.get("OPENAI_ENGINE_ID", "gpt-5.2")


def api_request(messages, engine, client, backend, **kwargs):
    """
    Calls the underlying LLM endpoint based on the backend.
    Retries with exponential backoff on transient errors (rate limits, server errors).
    """
    max_retries = 10
    for attempt in range(max_retries):
        try:
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
        except Exception as e:
            wait = min(2 ** attempt * 2, 120)
            print(f"API request error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"API request failed after {max_retries} retries")

def api_infer_single(
        prompt,
        max_token_length=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        return_format="",
        system_prompt=None,):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
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
        return_format="",
        system_prompt=None,):
    
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
            return_format=return_format,
            system_prompt=system_prompt,
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
        



