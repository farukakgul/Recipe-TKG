from config import parse_args
from utils import get_lora_config, get_model_and_tokenizer, process_data, get_trainer, get_dataset
import wandb
import re
from datetime import datetime
import os
from datasets import concatenate_datasets 


def set_wandb_para(proj_name):
    os.environ["WANDB_PROJECT"] = proj_name # name your W&B project 
    os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB__SERVICE_WAIT"] = "300" #prevent issue https://github.com/wandb/wandb/issues/3911

def convert_to_llama3_prompt(example):
    """Convert Llama2 prompts to Llama3 format"""
    ins = '''<|begin_of_text|>
                You must be able to correctly predict the next {object_label} from \
                a given text consisting of multiple quadruplets in the form of "{time}:[{subject}, {relation}, {object_label}.{object}]" \
                and the query in the form of "{time}:[{subject}, {relation}," in the end.\n\
                You must generate {object_label}.{object}\n\n'''
    # Extract components from original prompt
    match = re.search(r'<<SYS>>(.*?)<</SYS>>(.*?)\[/INST\]', example["context"], re.DOTALL)
    if match:
        history_part = match.group(2).strip()
        new_prompt = ins + '\n' + history_part
        return {"context": new_prompt, "target": f"{example['target']}<|eot_id|>" }
    return example

def prepare_data(args, tokenizer):
    # Load the main 1024 samples
    main_dataset_dict = get_dataset(args.DATA_TYPE, args.DATA_PATH)
    main_dataset = main_dataset_dict['train']

    # Load the large dataset and take random 10k samples if provided
    if args.FULL_DATA_PATH != "":
        large_dataset = get_dataset(args.DATA_TYPE, args.FULL_DATA_PATH)
        large_train = large_dataset['train']
        if len(large_train) > args.APPEND_DATA_SIZE:
            large_train = large_train.shuffle(seed=42).select(range(args.APPEND_DATA_SIZE))
        
        # Combine datasets
        main_dataset = concatenate_datasets([
            main_dataset,
            large_train
        ])
    if "Llama-3" in args.MODEL_NAME:
        main_dataset = main_dataset.map(
            convert_to_llama3_prompt,
            load_from_cache_file=False
        )
    # Process the combined data
    processed_data = process_data(args, tokenizer, args.DATA_TYPE, main_dataset)

    # Evaluation data
    if args.EVAL_PATH is not None and args.EVAL_STRATEGY != "no":
        if args.EVAL_BY_HF:
            with open(args.EVAL_PATH, 'r', encoding='utf-8') as file:
                content = file.read()
            eval_content = content.split('\n\n')
            eval_dataset = Dataset.from_dict({'text': eval_content})
        else:
            eval_dataset = get_dataset(args.EVAL_TYPE, args.EVAL_PATH)['train']
        
        eval_data = process_data(args, tokenizer, args.EVAL_TYPE, eval_dataset)
        return {"train": processed_data, "test": eval_data}
    else:
        return {"train": processed_data}

def generate_run_name(outdir, time=None):
    match = re.search(r'([^\/]+)$', outdir)
    filename = match.group(1) if match else ""
    run_name = filename + "_"
    if time is None:
        run_name = run_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_name = run_name + time
    return run_name

if __name__ == "__main__":
    args = parse_args()
    # os.environ["WANDB_MODE"] = "disabled"
    if args.REPORT_TO == "wandb":
        #init to wandb
        wandb.login()
        if args.RUN_NAME is None:
                args.RUN_NAME = generate_run_name(args.OUTPUT_DIR)
                print("gen_name", args.RUN_NAME)
        #args to wandb
        set_wandb_para(args.PROJ_NAME)

    print("load_best_model_at_end is set as", args.LOAD_BEST_MODEL_AT_END)

    lora_config = get_lora_config(args)
    model, tokenizer = get_model_and_tokenizer(args, lora_config)

    data = prepare_data(args, tokenizer)
    #wandb is combined with transformers.Trainer, by using TrainingArguments, as shown in the official site
    if args.W_RESUME == 1:
        trainer = get_trainer(args, model, data, tokenizer) 
        trainer.train(resume_from_checkpoint=args.RESUME_CKPT)
        trainer.save_model(args.OUTPUT_DIR + "/model_final")
    else:
        trainer = get_trainer(args, model, data, tokenizer) 
        trainer.train(resume_from_checkpoint=False)
        trainer.save_model(args.OUTPUT_DIR + "/model_final")
