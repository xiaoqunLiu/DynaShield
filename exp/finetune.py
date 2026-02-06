import numpy as np
import os
import time, random

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import sys
import re
import json
import copy
import torch
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.opt_utils import load_model_and_tokenizer
from utils.string_utils import PromptManager, load_conversation_template
from utils.generate import generate
from utils.model import GPT



def get_args():
    parser = argparse.ArgumentParser(description="Finetune manager.")
    # Experiment Settings
    parser.add_argument("--model_name", type=str, default="llama3")

    # Finetune (Generation) Parameters
    parser.add_argument("--top_p", type=int, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num_trials", type=int, default=2)
    parser.add_argument("--max_trials", type=int, default=5)

    # Finetune (LoRa) Parameters
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--bias", type=str, default="none")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_seq_length", type=int, default=2048)
   
    # System Settings
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--BF16", type=bool, default=True)
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=True)
    parser.add_argument("--use_cache", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--GPT_API", type=str, default=None)

    return parser.parse_args()

args = get_args()

# API Key
# if args.GPT_API is None:
#     raise ValueError("GPT_API is required for GPT check.")

# Set the random seed for NumPy
np.random.seed(args.seed)
# Set the random seed for PyTorch
torch.manual_seed(args.seed)
# If you are using CUDA (i.e., a GPU), also set the seed for it
torch.cuda.manual_seed_all(args.seed)

# Load model and template
if args.model_name == "llama3":
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    template_name = 'llama-3'
elif args.model_name == "dolphin":
    model_name = "dphn/dolphin-2.9.1-llama-3-8b"
    template_name = "chatml"
elif args.model_name == "hermes":
    model_name = "NousResearch/Hermes-3-Llama-3.1-8B"
    template_name = 'chatml'
elif args.model_name == "userlm":
    model_name = "microsoft/UserLM-8b"
    template_name = "chatml"
elif args.model_name == "guard":
    model_name = "meta-llama/Llama-Guard-3-8B"
    template_name = "chatml"
elif args.model_name == "bagel":
    model_name = "jondurbin/bagel-8b-v1.0"
    template_name = 'llama-3.1'
elif args.model_name == "llama32":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    template_name = 'llama-3.1'
elif args.model_name == "stor":
    model_name = "akjindal53244/Llama-3.1-Storm-8B"
    template_name = 'llama-3.1'
elif args.model_name == "llama2":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    template_name = 'llama-2'
else:
    raise ValueError("Invalid model name. If you want to use other models, please use the main branch.")

# Logging Settings
output_dir = "../lora_modules/" + args.model_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

log_name = "finetune_"+args.model_name+".log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, log_name)),
        logging.StreamHandler()
    ]
)
logging.info(f"Args: {args}")

# Load Model, Tokenizer and Template
device = f'cuda:{args.device}'
model, tokenizer = load_model_and_tokenizer(model_name, 
                       BF16=args.BF16,
                       low_cpu_mem_usage=args.low_cpu_mem_usage,
                       use_cache=args.use_cache,
                    #    do_sample=False,
                       device=device)

conv_template = load_conversation_template(template_name)
if model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
    conv_template.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
elif model_name == "dphn/dolphin-2.9.1-llama-3-8b":
    conv_template.system_message = (
        "You are Dolphin, a helpful and harmless AI assistant. "
        "You always respond helpfully, truthfully, and refuse unsafe or unethical requests."
        "You respond with clarity, warmth, and curiosity. "
        "Be helpful, honest, and safe — avoid harmful or biased content. "
        "Explain ideas simply, ask clarifying questions when needed, and keep your tone positive and professional."
    )
elif model_name == "NousResearch/Hermes-3-Llama-3.1-8B":
    conv_template.system_message = (
        "You are Hermes 3, a helpful and harmless assistant based on LLaMA 3.1. "
        "Provide accurate, thoughtful, and ethical answers. "
        "Refuse unsafe, illegal, or unethical requests."
    )
elif model_name == "microsoft/UserLM-8b":
    conv_template.system_message = (
        "You are UserLM, a helpful, safe, and responsible assistant developed by Microsoft. "
        "You must follow ethical guidelines and refuse harmful or unsafe instructions."
    )
elif model_name == "meta-llama/Llama-Guard-3-8B":
    conv_template.system_message = (
        "You are Llama Guard 3, a safety and moderation assistant based on LLaMA 3. "
        "Your role is to identify, prevent, or politely refuse unsafe, illegal, or harmful content. "
        "Always provide concise, responsible, and well-reasoned answers."
    )
elif model_name == "jondurbin/bagel-8b-v1.0":
    conv_template.system_message = (
        "You are Bagel, a friendly and intelligent AI assistant trained by Jondurbin. "
        "You are helpful, honest, and harmless. "
        "Provide thoughtful, well-reasoned answers. "
        "When faced with unsafe, unethical, or illegal requests, you must refuse politely "
        "and explain why the request cannot be fulfilled."
    )
elif model_name == "meta-llama/Llama-3.2-3B-Instruct":
    conv_template.system_message = (
    "You are a thoughtful, reliable, and compassionate assistant. "
    "Respond with clarity and depth, aiming to genuinely help the user while staying responsible and safe. "
    "Encourage understanding, fairness, and positive engagement in every reply. "
    "If a question could lead to harm, respond with care—explain the concern and guide the conversation toward something constructive and respectful."
)
elif model_name == "akjindal53244/Llama-3.1-Storm-8B":
    conv_template.system_message = (
        "You are a capable, balanced, and insightful assistant. "
        "Approach every query with clarity and reasoning, offering accurate and thoughtful answers. "
        "Encourage understanding and honesty, and always prioritize factual correctness over speculation. "
        "If a request seems unsafe, unclear, or potentially harmful, respond with care and guide the user toward a responsible direction."
    )
elif model_name == "meta-llama/Llama-2-7b-chat-hf":
    conv_template.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
logging.info(f"System Prompt: {conv_template.system_message}")

system_prompt = conv_template.system_message
gen_config = model.generation_config
gen_config.max_new_tokens = args.max_new_tokens
gen_config.do_sample = True
gen_config.top_p = args.top_p
gen_config.temperature = args.temperature
num_trials = args.num_trials
max_trials = args.max_trials
logging.info(f"Generation Config: {gen_config}")

ft_datasets = []
save_path = output_dir + "/ft_datasets_"+args.model_name+".json"

# Load naive harmful prompts
with open('../datasets/seed_reject.json', 'r', encoding='utf-8') as file:
    seed_reject = json.load(file)

attack_prompts = [prompt["prompt"] for prompt in seed_reject["prompts"]]
logging.info(f"Number of attack prompts: {len(attack_prompts)}")

# Detection Model
# detection_model = GPT('gpt-4', api=args.GPT_API)

# for user_prompt in tqdm(attack_prompts):
#     prompt_manager = PromptManager(tokenizer=tokenizer, 
#             conv_template=conv_template, 
#             instruction=user_prompt,
#             verbose=False)

#     input_ids = prompt_manager.get_input_ids().to(device)
#     logging.info(f"Full inputs: {tokenizer.decode(input_ids)}")

#     for trial in range(num_trials):
#         valid_completion = None
#         is_success = False
#         for trial_idx in range(max_trials):
#             completion = tokenizer.decode((generate(model, tokenizer, input_ids, gen_config=gen_config))).strip()
#             logging.info(f"\nCompletion: {completion}")

#             completion_rejection_detection = f"Please detect if the following answer explicitly rejects the user question at the beginning:\nUser question: {user_prompt}\nAnswer: {completion}. Reply with 'Yes' or 'No'."
#             time.sleep(random.uniform(1.0, 2.0)) 
#             detection_output = detection_model(completion_rejection_detection).choices[0].message.content
#             logging.info(f"Detection Output: {detection_output}")

#             if len(completion) < args.min_new_tokens:
#                 is_success = False
#                 logging.info("Completion too short. Re-generating completion.")
#             elif "yes" in detection_output.lower():
#                 is_success = True
#                 logging.info("Completion explicitly rejects user question. Saving completion.")
#                 valid_completion = completion
#                 break
#             else:
#                 is_success = False
#                 logging.info("Completion does not explicitly reject user question. Re-generating completion.")
        
#         if is_success:
#             # Remove system prompt from prompt to get user prompt
#             if template_name == "llama-3":
#                 full_prompt = tokenizer.decode(input_ids)
#                 logging.info(f"Full Prompt: {full_prompt}")
#                 llama3_system_pattern = r'<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\s*.*?<\|eot_id\|>'
#                 user_prompt = re.sub(llama3_system_pattern, '', full_prompt, flags=re.DOTALL)
#                 saved_prompt = user_prompt + valid_completion
#                 ft_datasets.append({'text': saved_prompt})
#                 logging.info(f"Saved: {saved_prompt}")
#             elif template_name == "chatml":
#                 full_prompt = tokenizer.decode(input_ids)
#                 logging.info(f"Full Prompt: {full_prompt}")
#                 # remove the system and assistant parts, keep the user message
#                 chatml_pattern = r'<\|im_start\|>system.*?<\|im_end\|>'
#                 user_prompt = re.sub(chatml_pattern, '', full_prompt, flags=re.DOTALL).strip()
#                 saved_prompt = user_prompt + " " + valid_completion
#                 ft_datasets.append({'text': saved_prompt})
#                 logging.info(f"Saved: {saved_prompt}")
#             elif template_name == "llama-3.1":
#                 full_prompt = tokenizer.decode(input_ids)
#                 logging.info(f"Full Prompt: {full_prompt}")
#                 # Remove the system message (everything between system header and its end-of-turn)
#                 llama31_system_pattern = (
#                     r'<\|begin_of_text\|>?\s*'
#                     r'<\|start_header_id\|>system<\|end_header_id\|>'
#                     r'.*?<\|eot_id\|>'
#                 )
#                 # Strip the system section from the conversation
#                 user_prompt = re.sub(llama31_system_pattern, '', full_prompt, flags=re.DOTALL).strip()
#                 # Combine with valid completion
#                 saved_prompt = user_prompt + " " + valid_completion
#                 ft_datasets.append({'text': saved_prompt})
#                 logging.info(f"Saved: {saved_prompt}")

#             else:
#                 raise ValueError("Invalid template name. If you want to use other models, please use the main branch.")

# with open(save_path, 'w', encoding='utf-8') as f:
#     json.dump(ft_datasets, f, ensure_ascii=False, indent=4)


# LoRa Training
# Load Dataset
dataset = load_dataset('json', data_files=save_path, split="train")

# Define LoRA parameters
peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_r,
    bias=args.bias,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

from trl import SFTConfig

training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optim=args.optim,
    num_train_epochs=args.num_train_epochs,
    logging_steps=args.logging_steps,
    learning_rate=args.learning_rate,
    fp16=False,
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=args.warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=args.lr_scheduler_type,
)

# Set tokenizer and SFT-specific parameters
training_arguments.processing_class = tokenizer
training_arguments.dataset_text_field = "text"
training_arguments.max_seq_length = args.max_seq_length
training_arguments.packing = False

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_arguments,
)

trainer.train()

# Debug: Check if LoRa B Matrix is 0
lora_params = {n: p for n, p in model.named_parameters() if "lora_B" in n}
if next(iter(lora_params.values())).any():
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    logging.info(f"Model is saved to {output_dir}. All done!")
else:
    logging.info("LoRA B Matrix is 0. Please Debug. Model not saved.")