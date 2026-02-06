export CUDA_VISIBLE_DEVICES=xxx

# generate files
for model in gemma; do
    python mtd.py --model_name $model  --disable_GPT_judge
done

#Lora finetuning
for model in llama2; do
    python finetune.py --model_name $model --GPT_API "$OPENAI_API_KEY"
done

for model in llama3; do
    for attack in GCG AutoDAN PAIR DeepInception; do
        for defense in Self-Reminder Self-Exam SafeDecoding MTDP MTDD MTDDP MTDBP MTDBD; do
            python defense.py --model_name $model --attacker $attack --defender $defense --disable_GPT_judge
        done
    done
done  

for model in llama3; do
    for attack in GCG AutoDAN PAIR DeepInception MTDP MTDD MTDDP MTDBP MTDBD; do
        python defense.py --model_name $model --attacker $attack --defender SmoothLLM --smoothllm_pert_type swap   --smoothllm_pert_pct 10   --smoothllm_num_copies 5   --max_new_tokens 100 --disable_GPT_judge --device 0 >> defense_$model.log 2>&1
    done
done