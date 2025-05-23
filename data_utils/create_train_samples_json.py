# Create a json file of train dataset from the txt files
# data/processed/DATASET/{train,valid} -> data/processed/DATASET/train_samples/DATASET_{sampling_type}.json

import json
import os

dataset = "yago"
retrieve_type = "rbmh"

with open(f"../data/processed/train/{dataset}/raw/train/history_facts_{dataset}_{retrieve_type}.txt", "r", encoding="utf-8") as f:
    train_data = f.read()
    train_data  = train_data.strip().split('\n\n')
    train_data = [e.strip() for e in train_data if e.strip()]

with open(f"../data/processed/train/{dataset}/raw/valid/history_facts_{dataset}_{retrieve_type}.txt", "r", encoding="utf-8") as f:
    valid_data = f.read()
    valid_data  = valid_data.strip().split('\n\n')
    valid_data = [e.strip() for e in valid_data if e.strip()]

with open(f"../data/processed/train/{dataset}/raw/train/test_ans_{dataset}.txt", "r", encoding="utf-8") as f:
    train_answer = f.readlines()

with open(f"../data/processed/train/{dataset}/raw/valid/test_ans_{dataset}.txt", "r", encoding="utf-8") as f:
    valid_answer = f.readlines()

with open(F"../data/original/{dataset}/entity2id.json", "r", encoding="utf-8") as f:
    entity2id = json.load(f)

prompt = "<s>[INST] <<SYS>>You must be able to correctly predict the next {object_label} from a given text consisting of multiple quadruplets in the form of \"{time}:[{subject}, {relation}, {object_label}.{object}]\" and the query in the form of \"{time}:[{subject}, {relation},\" in the end.\nYou must generate {object_label}.{object}\n\n<</SYS>>"

all_training_sample = []

for i, sample in enumerate(train_data):
    history = sample.strip().split('\n')[:-1]
    history_str = " \n".join(h.strip() for h in history)
    query = sample.strip().split('\n')[-1]
    answer = train_answer[i].strip().split('\t')[2]
    answer_id = entity2id[answer]
    if history_str == "":
        train_input = prompt + query + "[/INST]"
    else:
        train_input = prompt + history_str + " \n" + query + "[/INST]"
    train_label = str(answer_id) + "." + answer
    all_training_sample.append({"context": train_input, "target": train_label})

for i, sample in enumerate(valid_data):
    history = sample.strip().split('\n')[:-1]
    history_str = "\n".join(h.strip() for h in history)
    query = sample.strip().split('\n')[-1]
    answer = valid_answer[i].strip().split('\t')[2]
    answer_id = entity2id[answer]
    if history_str == "":
        valid_input = prompt + query + "[/INST]"
    else:
        valid_input = prompt + history_str + " \n" + query + "[/INST]"
    valid_label = str(answer_id) + "." + answer
    all_training_sample.append({"context": valid_input, "target": valid_label})

os.makedirs(f"../data/processed/train/{dataset}/raw/train_samples", exist_ok=True)

with open(f"../data/processed/train/{dataset}/raw/train_samples/{dataset}_{retrieve_type}.json", "w", encoding="utf-8") as f:
    json.dump(all_training_sample, f, indent=4)
