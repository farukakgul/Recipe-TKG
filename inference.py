import torch
import re
import time

import os
from evaler import Evaler
from eval_utils import parse_args, read_test_and_divide, read_test_an, read_last_metric, decide_model

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    args = parse_args()

    parent_dir = os.path.dirname(args.output_file)
    eval_txt_path = parent_dir+'/metric_results.txt'
    
    dirpath = os.path.dirname(args.output_file) 
    os.makedirs(dirpath, exist_ok=True)

    test_ans = read_test_an(args.test_ans_file)
    if args.ft == 1:
        while not os.path.exists(args.LORA_CHECKPOINT_DIR):
            print(f"Waiting for the fine-tuned model to be available at {args.LORA_CHECKPOINT_DIR}...")
            time.sleep(1200)  # Wait for 20 minutes before checking again    
        model = decide_model(args)      
        print('Using finetuned model ')
    else:
        if args.BIT_4:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.MODEL_NAME,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.MODEL_NAME,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        print('Using base model (not a fine-tuned one).')

    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token   
    tokenizer.padding_side = "left"
    tests = read_test_and_divide(args.input_file)

    c = read_last_metric(args.last_metric,args.apply_semantic_filter) #dict with keys: c1 c3 c10

    if "Llama-3" in args.MODEL_NAME:
        patterns = [
        # Match answers in format: <|start_header_id|>assistant<|end_header_id|>\n\nANSWER<|eot_id|>
        re.compile(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)', re.DOTALL),
        # Match answers with numeric prefixes and optional brackets
        re.compile(r'\b(?:\d+[.:])?[\s_]*([\w\-()\/&,\.\'\s]+?)(?:[\]\[<]|$|,)'),
        # Fallback pattern for any remaining artifacts
        re.compile(r'[^a-zA-Z0-9()\-_&\/\'\s]')  # Remove special characters
        ]

        early_stop_chars = [
           torch.tensor([60], device='cuda:0'),      # ]
            torch.tensor([8], device='cuda:0')        # )
        ]
    
    else:
        pattern1 = re.compile(r'.*?[\d:@][._](.*?)[\]\[]?([< ].*?)?$') 
        pattern2 = re.compile(r'<s> .*?[\n]?([A-Z\u00C0-\u00DD\u0388-\u03AB\u0410-\u042F\u0600-\u06FF\u4e00-\u9fa5].*)\]')
        pattern3 = re.compile(r'<s> *(.*)\]') 
        is_with_id = True
        if is_with_id:
            patterns = [pattern1]
        else:
            patterns = [pattern1, pattern2, pattern3]

        early_stop_chars = [torch.tensor([29962], device='cuda:0'), #]
                        torch.tensor([29961], device='cuda:0'), #[
                        torch.tensor([4638], device='cuda:0'),  #)]
                        torch.tensor([29871], device='cuda:0')]  # 

    topk= 10
    cnt = args.begin 
    obligations = []
    evaler = Evaler(topk, tests, test_ans, eval_txt_path, args, model, tokenizer, patterns, early_stop_chars, obligations)
    
    path_results = args.path_results
    path_results = os.path.normpath(path_results)

    if path_results != '.':
        if args.apply_semantic_filter:
            evaler.eval_filtering(c, cnt, path_results)
        else:
            evaler.eval(c, cnt, path_results)
    else:
        if args.apply_semantic_filter:
            evaler.eval_filtering(c, cnt, filter_yes=args.FILTER)
        else:
            evaler.eval(c, cnt, filter_yes=args.FILTER)