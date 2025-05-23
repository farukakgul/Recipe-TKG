# Sample few-shot datasets from a train dataset. Align with TLR few-shot datasets.
# data/processed/DATASET/train_samples/DATASET_{sampling_type}.json" -> data/processed/DATASET/train_samples/DATASET_{sampling_type}_{samping_num}.json  if align = False
# data/processed/DATASET/train_samples/DATASET_{sampling_type}.json" -> data/processed/DATASET/train_samples/DATASET_{sampling_type}_{samping_num}_align.json  if align = True

import json
import random
from collections import defaultdict
import unicodedata

dataset = 'yago'
retrieve_type = 'rbmh'
align = True
retrieve_num = 1024

def normalize_unicode(text):
    if isinstance(text, str):
        return unicodedata.normalize('NFC', text).strip().lower()
    return text

input_file = f"../data/processed/train/{dataset}/raw/train_samples/{dataset}_{retrieve_type}.json"
with open(input_file, "r", encoding="utf-8") as f:
    all_samples = json.load(f)

# all_samples = normalize_json(all_samples)

def get_match_key(context):
    lines = context.split('\n')
    if len(lines) == 4:
        return lines[-1].split('<</SYS>>')[-1]
    return lines[-1]

if align:
    alignment_file = f"../data/TLR/train/{dataset}/{dataset}_{retrieve_num}.json"
    with open(alignment_file, "r", encoding="utf-8") as f:
        alignment_json = json.load(f)
    # alignment_json = normalize_json(alignment_json)

    sample_dict = defaultdict(list)
    for sample in all_samples:
        key = (sample['target'].split('.')[0], normalize_unicode(get_match_key(sample['context'])))
        sample_dict[key].append(sample)

    matched = []
    for few_item in alignment_json:
        key = (few_item['target'].split('.')[0], normalize_unicode(get_match_key(few_item['context'])))
        if key in sample_dict:
            matched.append(sample_dict[key][0])
        else:
            print("Not found")
            # print(len(few_item['context'].split('\n')))
            # print(repr(few_item['context'].split('\n')[-1]))
            # print(repr(few_item['target']))
            # print(few_item)


    output_file = f"../data/processed/train/{dataset}/{dataset}_{retrieve_type}_{retrieve_num}_align.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(matched, json_file, indent=4)

else:
    random.seed(42)
    sampled_data = random.sample(all_samples, retrieve_num)
    output_file = f"../data/processed/train/{dataset}/{dataset}_{retrieve_type}_{retrieve_num}.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(sampled_data, json_file, indent=4)
