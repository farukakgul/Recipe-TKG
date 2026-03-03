#!/bin/bash

# An example: yago-ft-tlr
CUDA_VISIBLE_DEVICES=1 \
python3 inference.py \
  --output_file "./output/yago_llama2_ft_tlr/final.txt" \
  --input_file "./data/processed/eval/yago/history_facts_yago_tlr.txt" \
  --test_ans_file "./data/processed/eval/yago/test_ans_yago.txt" \
  --MODEL_NAME "meta-llama/Llama-2-7b-hf" \
  --ft 1 \
  --LORA_CHECKPOINT_DIR "./model/yago_llama2_ft_tlr/model_final/" \
  
