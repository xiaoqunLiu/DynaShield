import gc
import logging
import subprocess
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.model import get_conversation_template

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
        model_path,
        BF16=True,
        tokenizer_path=None,
        device='cuda:0',
        **kwargs
):
    """
    Universal loader for LLaMA, LLaMA-3.1, Dolphin, Hermes, Bagel, Storm, UserLM, Gemma-3.

    Key guarantees:
    - For large models (especially Gemma-3), ALWAYS use device_map="auto"
      and NEVER call .to(device).
    - Remove incompatible arguments (use_cache, low_cpu_mem_usage) for models that don't support them.
    - LoRA will later be injected correctly based on model.hf_device_map.
    """

    # ---------- Clean kwargs ----------
    # Gemma-3 does NOT allow use_cache in from_pretrained
    if "use_cache" in kwargs:
        kwargs.pop("use_cache")

    # Some models dislike low_cpu_mem_usage=True
    if "low_cpu_mem_usage" in kwargs:
        kwargs.pop("low_cpu_mem_usage")

    # ----------------------------------
    # Set dtype
    # ----------------------------------
    load_kwargs = dict(trust_remote_code=True, **kwargs)
    if BF16:
        load_kwargs["torch_dtype"] = torch.bfloat16

    # ----------------------------------
    # Critical: ALWAYS device_map="auto" for large models
    # ----------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",     # <<< accelerate will place layers on multiple GPUs
        **load_kwargs
    )

    # ------------------------------
    # DO NOT MOVE MODEL MANUALLY!!!
    # ------------------------------
    # accelerate

    model = model.eval()

    # ----------------------------------
    # Load tokenizer
    # ----------------------------------
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# def load_model_and_tokenizer(model_path, BF16 = True, tokenizer_path=None, device='cuda:0', **kwargs):
#     """
#     Load a causal LM and its tokenizer.

#     Notes:
#     - If caller passes device_map or load_in_8bit in kwargs, we do NOT call .to(device)
#       so that the transformers/device_map logic can place parameters appropriately.
#     - Otherwise, we load the model and move it to `device` (e.g. 'cuda:0').
#     """
#     # Choose dtype arg when BF16 requested
#     from_pretrained_kwargs = dict(trust_remote_code=True, **kwargs)
#     if BF16:
#         from_pretrained_kwargs['torch_dtype'] = torch.bfloat16

#     # If user requested device_map or 8-bit loading, let transformers handle placement
#     handle_placement = not (('device_map' in from_pretrained_kwargs) or ('load_in_8bit' in from_pretrained_kwargs))

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         **from_pretrained_kwargs
#     )

#     # if handle_placement and device is not None:
#     #     # move model to requested device
#     #     model = model.to(device)

#     model = model.eval()

#     tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_path,
#         trust_remote_code=True,
#         use_fast=False
#     )

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     return model, tokenizer

def get_latest_commit_info():
    try:
        # Get the latest commit hash
        commit_hash = subprocess.run(["git", "log", "-1", "--format=%H"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Get the latest commit date
        commit_date = subprocess.run(["git", "log", "-1", "--format=%cd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if both commands were executed successfully
        if commit_hash.returncode == 0 and commit_date.returncode == 0:
            return commit_hash.stdout.strip(), commit_date.stdout.strip()
        else:
            error_message = commit_hash.stderr if commit_hash.returncode != 0 else commit_date.stderr
            return "Error fetching commit information:", error_message
    except FileNotFoundError:
        # Git not installed or not found in the path
        return "Git is not installed or not found in the path.", ""