# partition all_facts into train, valid and test set
# data/original/DATASET/all_facts.txt  |lexical format| -> data/original/DATASET/{train,valid,test}.txt  |id format|
#                                                       -> data/processed/DATASET/{train,valid,test}.txt  |lexical format|

import json

# train valid test
STATISTICS = {"icews14": [74845, 8514, 7371],
              "icews18": [373018, 45995, 49545],
              "icews0515": [368868, 46302, 46159],
              "GDELT": [79319, 9957, 9715],
              "YAGO": [220393, 28948, 22765]}

dataset = "GDELT"

with open(f"../data/original/{dataset}/relation2id.json", "r", encoding="utf-8") as f:
    relation2id = json.load(f)
with open(f"../data/original/{dataset}/entity2id.json", "r", encoding="utf-8") as f:
    entity2id = json.load(f)
with open(f"../data/original/{dataset}/ts2id.json", "r", encoding="utf-8") as f:
    ts2id = json.load(f)

facts_dir = f"../data/original/{dataset}/all_facts.txt"
with open(facts_dir, "r", encoding="utf-8") as f:
    facts = f.readlines()

train_facts = facts[:STATISTICS[dataset][0]]
valid_facts = facts[STATISTICS[dataset][0]:STATISTICS[dataset][0]+STATISTICS[dataset][1]]
test_facts = facts[STATISTICS[dataset][0]+STATISTICS[dataset][1]:]


# with open(f"../data/original/{dataset}/train.txt", "w", encoding="utf-8") as f:
#     for fact in train_facts:
#         f.write(str(entity2id[fact.split('\t')[0]]) + "\t" + str(relation2id[fact.split('\t')[1]]) + "\t" + str(entity2id[fact.split('\t')[2]]) + "\t" + str(ts2id[fact.split('\t')[3][:-1]]) + "\n")

# with open(f"../data/original/{dataset}/valid.txt", "w", encoding="utf-8") as f:
#     for fact in valid_facts:
#         f.write(str(entity2id[fact.split('\t')[0]]) + "\t" + str(relation2id[fact.split('\t')[1]]) + "\t" + str(entity2id[fact.split('\t')[2]]) + "\t" + str(ts2id[fact.split('\t')[3][:-1]]) + "\n")

# with open(f"../data/original/{dataset}/test.txt", "w", encoding="utf-8") as f:
#     for fact in test_facts:
#         f.write(str(entity2id[fact.split('\t')[0]]) + "\t" + str(relation2id[fact.split('\t')[1]]) + "\t" + str(entity2id[fact.split('\t')[2]]) + "\t" + str(ts2id[fact.split('\t')[3][:-1]]) + "\n")

with open(f"../data/processed/{dataset}/train.txt", "w", encoding="utf-8") as f:
    for fact in train_facts:
        f.write(fact)

with open(f"../data/processed/{dataset}/valid.txt", "w", encoding="utf-8") as f:
    for fact in valid_facts:
        f.write(fact)

with open(f"../data/processed/{dataset}/test.txt", "w", encoding="utf-8") as f:
    for fact in test_facts:
        f.write(fact)
