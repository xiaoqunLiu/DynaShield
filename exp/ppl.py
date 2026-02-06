import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
import torch
import subprocess
import argparse
from datasets import load_dataset, concatenate_datasets
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.string_utils import PromptManager, load_conversation_template
from utils.opt_utils import load_model_and_tokenizer, get_latest_commit_info
from utils.safe_decoding import SafeDecoding
from utils.ppl_calculator import PPL_Calculator
from utils.bpe import load_subword_nmt_table, BpeOnlineTokenizer
from utils.model import GPT
from safe_eval import DictJudge, GPTJudge
import numpy as np
from tqdm import tqdm
import copy, json, time, logging
from peft import PeftModel, PeftModelForCausalLM

result_path = '/result_path'
ppl_calculator = PPL_Calculator(model = 'gpt2')
import os
result_files = os.listdir(result_path)# result_files = [i.replace('MTDD','MTD_D') for i in result_files if not 'MTDP' in i and not 'MTDDP' in i]
# read the file end with safe_eval.json
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

all = []

for dir in result_files:
    r = {}
    p_all = []

    for item in os.listdir(os.path.join(result_path, dir)):
        if 'safe_eval.json' in item:
            # print(item)
            data = read_json(os.path.join(result_path, dir, item))
            # handle each json
            attack_success_count = 0 # attack score, asr +1  if safe_eval_results[i]['dic_score'] == 1: defense_success_count += 1
            attack_success = [dataitem['dic_score'] for dataitem in data]
            attack_success_count = sum(attack_success)
            # print(score/len(data))
            dir = dir.replace('MTD_D', 'MTDD')
            model = dir.split('_')[1]
            defense = dir.split('_')[0]
            attack = dir.split('_')[2]

            r['attack_success_count'] = 1- attack_success_count/len(data)
            r['model'] = model
            r['defense'] = defense
            r['attack'] = attack
            all.append(r)

            parameter = read_json(os.path.join(result_path, dir, item).replace('_safe_eval', ''))
            for dataitem in parameter['data']:
                p = {}
                if 'top_k' not in dataitem: continue
                p['model'] = model
                p['defense'] = defense
                p['attack'] = attack
                p['ppl_score'] =  ppl_calculator.get_perplexity(dataitem['output']),
                p['attack_success'] = attack_success[dataitem['id']]
                top_k = dataitem['top_k']
                p['top_k'] = top_k
                top_p = dataitem['top_p']
                p['top_p'] = top_p
                temperature = dataitem['temperature']
                p['temperature'] = temperature
                p['time_cost'] = dataitem['time_cost'] 
                if 'sys_prompt' not in dataitem: 
                    sys_prompt = 'None'
                else: sys_prompt = dataitem['SYS_prompt']
                p['sys_prompt'] = sys_prompt
                p_all.append(p)
    # dump the parameter p_all
    with open(os.path.join(result_path, dir, 'parameter.json'), 'w') as f:
        json.dump(p_all, f, indent=4)

