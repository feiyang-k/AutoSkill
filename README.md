# AutoSkill
Code repo for AutoSkill.

### SFT Training Configurations
/sft_configs/*

### Evaluation Pipeline
/evaltask-v6.py

> Example usage: python evaltask-v6.py --model meta-llama/Meta-Llama-3-8B-Instruct --tasks math500 aime1k gsm8k aime25 amc olympiad minerva --rollouts 256 --temperature 1.0 --gen_len 8000 --max_sample 500 --output_path l3b8-v-8k-t100 --full_logs True --sys_prompt rl_prompt --devices '0,1,2,3,5,6,7,8' --seed 42 --verbose True
