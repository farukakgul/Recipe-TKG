from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict
import numpy as np
import re
from collections import defaultdict
import bitsandbytes as bnb


def get_model_and_tokenizer(args, config):
    if args.BIT_8:
        model = AutoModelForCausalLM.from_pretrained(
            args.MODEL_NAME,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    elif args.BIT_4:
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
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

    if "Llama-3" in args.MODEL_NAME:
        tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.MODEL_NAME,
            trust_remote_code=True,
            pad_token="</s>"
        )

    model = get_peft_model(model, config)

    embeddings = model.get_input_embeddings()
    if isinstance(embeddings, nn.Embedding):
        embeddings.requires_grad_(True)
        embeddings.weight.requires_grad = True
    elif hasattr(embeddings, 'weight'):
        embeddings.weight.requires_grad = True
    else:
        print("Warning: Could not find trainable embedding layer")
    model.config.use_cache = False
    return model, tokenizer


def llama2_tokenizer(args, tokenizer, data_type, data_point):
    if data_type == "json":
        data_slice_source = tokenizer(
            data_point["context"],
            max_length=args.CONTEXT_LEN,
            padding="max_length",
            truncation=True
        )
        data_slice_target = tokenizer(
            data_point["target"],
            max_length=args.TARGET_LEN,
            padding=False,
            truncation=True
        )

        data_slice = {}
        data_slice['input_ids'] = data_slice_source['input_ids'] + data_slice_target['input_ids'] + [
            tokenizer.eos_token_id] + [2] * (args.TARGET_LEN - len(data_slice_target['input_ids']))
        data_slice['attention_mask'] = data_slice_source['attention_mask'] + data_slice_target['attention_mask'] + [
            1] + [0] * (args.TARGET_LEN - len(data_slice_target['input_ids']))
        data_slice['labels'] = [-100] * args.CONTEXT_LEN + data_slice_target['input_ids'] + [
            tokenizer.eos_token_id] + [-100] * (args.TARGET_LEN - len(data_slice_target['input_ids']))


    elif data_type == "txt":
        data_slice = tokenizer(
            data_point["text"],
            max_length=args.TEXT_LEN,
            padding="max_length",
            truncation=True
        )
        data_slice['input_ids'] = data_slice['input_ids'].extend([tokenizer.eos_token_id])
        data_slice['attention_mask'] = data_slice['attention_mask'].extend([1])

    return data_slice

def llama3_tokenizer(args, tokenizer, data_type, data_point):
    if data_type == "json":
        full_prompt = data_point["context"] + data_point["target"]
        tokenized = tokenizer(
            full_prompt,
            max_length=args.CONTEXT_LEN + args.TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Find answer start position
        answer_start = len(tokenizer.encode(data_point["context"]))
        labels = tokenized["input_ids"].clone()
        labels[:, :answer_start] = -100
        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": labels[0]
        }
    elif data_type == "txt":
        # Tokenizer returns dict with lists, not tensors
        tokenized = tokenizer(
            data_point["text"],
            max_length=args.TEXT_LEN,
            padding="max_length",
            truncation=True
        )
        # Append EOS manually (works with list-based inputs)
        tokenized["input_ids"].append(tokenizer.eos_token_id)
        tokenized["attention_mask"].append(1)
        return tokenized

def process_data(args, tokenizer, data_type, dataset):
    tokenizer_fn = llama3_tokenizer if "Llama-3" in args.MODEL_NAME else llama2_tokenizer
    data = dataset.shuffle().map(
        lambda data_point: tokenizer_fn(
            args,
            tokenizer,
            data_type,
            data_point
        ),
        batched=False
    )
    return data


def get_lora_config(args):
    config = LoraConfig(
        r=args.LORA_R,
        lora_alpha=args.LORA_ALPHA,
        lora_dropout=args.LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
        target_modules=(
            ["q_proj", "k_proj", "v_proj", "o_proj"]  # For Llama3
            if "Llama-3" in args.MODEL_NAME else
            ["q_proj", "v_proj"]  # Original for Llama2
        )
    )
    return config

class llama2_trainer(transformers.Trainer):
    def __init__(self, model, tokenizer=None, ckpt='', data_path='', **kwargs):
        kwargs['model'] = model
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs,num_items_in_batch=None,return_outputs=False):
        print(model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss)
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

class SelfAttentionPooling(nn.Module):
    def __init__(self, d_input):
        super(SelfAttentionPooling, self).__init__()
        self.d_input = d_input
        self.query = nn.Linear(d_input, d_input)
        self.key = nn.Linear(d_input, d_input)
        self.value = nn.Linear(d_input, d_input)
        self.softmax = nn.Softmax(dim=1)  # Softmax over the sequence length (L)

    def forward(self, x):
        x = x.to(dtype=self.query.weight.dtype)
        query = self.query(x) 
        key = self.key(x)      
        value = self.value(x)  
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_input)
        attention_weights = self.softmax(attention_scores)
        weighted_sum = torch.matmul(attention_weights, value)
        pooled_output = weighted_sum.mean(dim=1)       
        return pooled_output

class CustomLoss_comp(nn.Module):
    def __init__(self, temperature=0.5, alpha=0.5):
        super(CustomLoss_comp, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, anchor_embeds, positive_embeds, negative_embeds, outputs):
        # anchor_embed: (batch_size, embedding_dim)
        # positive_embed: (batch_size, history_len + 1, embedding_dim)
        # negative_embed: (batch_size, history_len + 1, embedding_dim)
        # llama_logits: (batch_size,num_custom_tokens)
        # llama_labels: (batch_size)

        # Ensure input tensors require gradients
        contrastive_loss = torch.tensor(0.0, device=anchor_embeds[0].device)
        no_CL = False
        for i in range(len(anchor_embeds)):
            try:
                anchor_embeds[i].requires_grad_(True)
                positive_embeds[i].requires_grad_(True)
                negative_embeds[i].requires_grad_(True)
            except:
                continue

            margin = 1.0
            anchor_expanded = anchor_embeds[i].unsqueeze(0)  # (1, embedding_dim)

            pos_dist = torch.sum((positive_embeds[i] - anchor_expanded) ** 2, dim=-1)  # (num_pos)
            neg_dist = torch.sum((negative_embeds[i] - anchor_expanded) ** 2, dim=-1)  # (num_neg)

            hardest_pos_dist, _ = torch.max(pos_dist, dim=-1)  # (1)
            hardest_neg_dist, _ = torch.min(neg_dist, dim=-1)  # (1)

            contrastive_loss += F.relu(hardest_pos_dist - hardest_neg_dist + margin)
        if contrastive_loss == torch.tensor(0.0, device=anchor_embeds[0].device):
            no_CL = True

        ce_loss = outputs.loss

        if no_CL:
            total_loss = ce_loss
        else:
            total_loss = self.alpha * contrastive_loss + (1 - self.alpha) * ce_loss
        return total_loss, contrastive_loss, ce_loss
    
class llama2_trainer_contrastive_comp(transformers.Trainer):
    def __init__(self, model, tokenizer=None, ckpt='', data_path='', weight=0.5, **kwargs):
        super().__init__(model=model, **kwargs)
        self.attn_pooling = SelfAttentionPooling(model.config.hidden_size).to(self.model.device)
        self.loss_fn = CustomLoss_comp(alpha=weight)
        self.tokenizer = tokenizer
        self.ckpt = ckpt
        self.data_path = data_path

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(input_ids=inputs["input_ids"],labels=inputs["labels"],output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        device = hidden_states.device
        self.attn_pooling.to(device)

        anchor_sub_embeds, positive_embeds, negative_embeds = get_sample_embeddings_entity_comp(self.attn_pooling, self.tokenizer, model, self.data_path, inputs)
        if anchor_sub_embeds is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_loss, contrastive_loss, ce_loss = self.loss_fn(anchor_sub_embeds, positive_embeds, negative_embeds, outputs)
        print(f"contrastive_loss: {contrastive_loss.item()}, ce_loss: {ce_loss.item()}, total_loss: {total_loss.item()}")
        return total_loss

    def create_optimizer(self):
        self.optimizer = bnb.optim.AdamW8bit(list(self.model.parameters()) + list(self.attn_pooling.parameters()), 
                                             lr=self.args.learning_rate,
                                             betas=(self.args.adam_beta1,self.args.adam_beta2),
                                             weight_decay=0.01)

        opt_ckpt_file = os.path.join(self.ckpt, "optimizer.pt") if self.ckpt else None

        if opt_ckpt_file and os.path.exists(opt_ckpt_file):
            try:
                optimizer_state = torch.load(opt_ckpt_file, map_location=self.model.device)
                self.optimizer.load_state_dict(optimizer_state)
                print("[LOG] Loaded optimizer state from checkpoint.")
            except Exception as e:
                print(f"[FATAL] Failed to load optimizer state: {e}")
                return None 
        else:
            print("[LOG] No valid optimizer checkpoint, using fresh optimizer.")

        print(self.optimizer)
        return self.optimizer

    def _wrap_model(self, *args, **kwargs):
        model = super()._wrap_model(*args, **kwargs)
        self.create_optimizer()
        return model

def load_dictionary_senti(in_path: str, file_name: str) -> Dict[str, str]:
    _dict = {}
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            line_split = line.split("\t")
            node = line_split[0]
            sentiment = line_split[1].strip()

            _dict[node] = sentiment
    return _dict

def truncate_and_decode(tokens, eos_token, tokenizer):
    batch_size, seq_len = tokens.shape
    eos_positions = (tokens == eos_token).int()
    eos_indices = torch.argmax(eos_positions, dim=1)

    truncated_sentences = []
    for i in range(batch_size):
        truncated_tokens = tokens[i, :eos_indices[i]].tolist()
        decoded_sentence = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        truncated_sentences.append(decoded_sentence)

    return truncated_sentences

def extract_bracketed_content(strings):
    all_results = []
    
    for s in strings:
        # 1. Extract content after \n\n<</SYS>>
        match = re.search(r'\n\n<</SYS>>(.*)', s, re.DOTALL)
        if match:
            s = match.group(1).strip()  # Take the matched part and strip leading/trailing whitespace
        
        # 2. Find all content within [ ] brackets
        brackets_content = re.findall(r'\[(.*?)\]', s)

        # 3. Treat the content of each [] as an independent list and split by ','
        # extracted_values = [[item.strip() for item in content.split(',')] for content in brackets_content]
        extracted_values = []
        for i, content in enumerate(brackets_content):
            first_comma = content.find(',')
            last_comma = content.rfind(',')
            if first_comma != -1 and last_comma != -1 and first_comma != last_comma:
                part1 = content[:first_comma].strip()
                part2 = content[first_comma + 1:last_comma].strip()
                part3 = content[last_comma + 1:].strip()
                extracted_values.append([part1, part2, part3]) 
            else:
                raise ValueError(f"Invalid content in brackets: {content}")

        all_results.append(extracted_values)
    
    return all_results

def extract_between_tags(texts:list):
    results = []
    for text in texts:
        results.append(text.split('<s>')[-1].strip().split('</s>')[0].strip().split('.', 1)[1])
    return results

def get_batch_embedding(aggregator, tokenizer, model, texts):
    # texts is a list of entities
    tokens = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True)
    # with torch.no_grad():
    #     embeddings = model.get_input_embeddings()(tokens['input_ids'].to(model.device))  # (batch_size, seq_len, embedding_dim)
    embeddings = model.get_input_embeddings()(tokens['input_ids'].to(model.device))  # (batch_size, seq_len, embedding_dim)
    aggregated_embeddings = aggregator(embeddings)  # (batch_size, embedding_dim)
    return aggregated_embeddings

def get_sample_embeddings_entity_comp(attn_pooling, tokenizer, model, data_path, inputs):    
    parent_dir = os.path.dirname(data_path)
    sentiment_dictionary = load_dictionary_senti(parent_dir, "relation2sentiment.txt")

    truncated_inputs = truncate_and_decode(inputs["input_ids"], tokenizer.eos_token_id, tokenizer)
    # print('truncated_inputs: ', truncated_inputs)

    inputs_list = extract_bracketed_content(truncated_inputs)

    targets = extract_between_tags(tokenizer.batch_decode(inputs["input_ids"].tolist(), skip_special_tokens=False)) # [batch_size]
    # print('targets: ', targets)

    new_inputs_list = []  # [[s,r,o],[s,r,o],...,[s,r,o]]
    for i, b in enumerate(inputs_list):
        for a in b[:-1]:
            try:
                s, r, o = a
                new_inputs_list.append([s,r,o.split('.', 1)[1]])
            except:
                continue
        q_s, q_r, _ = b[-1]
        new_inputs_list.append([q_s, q_r, targets[i]])    

    groups = defaultdict(lambda: {"positive": [], "negative": []})
    for s, r, o in new_inputs_list:
        try:
            if sentiment_dictionary[r] == "positive":
                groups[s]["positive"].append(o)
            elif sentiment_dictionary[r] == "negative":
                groups[s]["negative"].append(o)
        except:
            continue

    # excluding the conflict items
    for s in groups:
        positive_set = set(groups[s]["positive"])
        negative_set = set(groups[s]["negative"])
        conflict_items = positive_set & negative_set
        groups[s]["positive"] = [o for o in groups[s]["positive"] if o not in conflict_items]
        groups[s]["negative"] = [o for o in groups[s]["negative"] if o not in conflict_items]

    groups = dict(groups)

    # Precompute embeddings for all events in the batch
    all_entities = [event[0]for event in new_inputs_list] + [event[2] for event in new_inputs_list]
    # all_embeddings = {event: get_single_embedding(attn_pooling, tokenizer, model, str(event), args.device) for event in set(all_events)}
    # print('all_entities: ', all_entities)

    unique_entities = list(set(all_entities))
    entity_texts = [str(entity) for entity in unique_entities]
    entity_embeddings = get_batch_embedding(attn_pooling, tokenizer, model, entity_texts)
    all_embeddings = {entity: entity_embeddings[i] for i, entity in enumerate(unique_entities)} # {entity: embedding, entity: embedding, ...}

    total_anchor_sub_embeds = []  # (group_size, embedding_dim)
    total_positive_embeds = []  # (group_size, num_pos, embedding_dim)
    total_negative_embeds = []  # (group_size, num_neg, embedding_dim)
    if len(groups) == 0:
        return None, None, None
    for subject, object_dict in groups.items():
        positive_embeds = []
        negative_embeds = []
        anchor_sub_embed = all_embeddings[subject]
        pos_entities = object_dict["positive"] # list
        neg_entities = object_dict["negative"]
        for entity in pos_entities:
            positive_embeds.append(all_embeddings[entity])
        for entity in neg_entities:
            negative_embeds.append(all_embeddings[entity])
        
        total_anchor_sub_embeds.append(anchor_sub_embed)
        if len(positive_embeds) == 0:
            total_positive_embeds.append(None)
        else:
            total_positive_embeds.append(torch.stack(positive_embeds))
        if len(negative_embeds) == 0:
            total_negative_embeds.append(None)
        else:
            total_negative_embeds.append(torch.stack(negative_embeds))
    return total_anchor_sub_embeds, total_positive_embeds, total_negative_embeds


def get_trainer(args, model, data, tokenizer):
    GRADIENT_ACCUMULATION_STEPS = args.BATCH_SIZE // args.MICRO_BATCH_SIZE
    LOAD_BEST_MODEL_AT_END = False
    if args.LOAD_BEST_MODEL_AT_END == 1:
        LOAD_BEST_MODEL_AT_END = True
    if args.CONTRASTIVE == 0:
        trainer = llama2_trainer(
        model=model,
        train_dataset=data['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=0.05,
            num_train_epochs=args.EPOCHS,
            learning_rate=args.LEARNING_RATE,
            save_strategy="steps",
            save_steps=args.SAVE_STEPS,
            eval_steps=args.EVAL_STEPS,
            output_dir=args.OUTPUT_DIR,
            overwrite_output_dir=True,
            save_total_limit=args.SAVE_TOTAL_LIMIT,
            evaluation_strategy=args.EVAL_STRATEGY,
            report_to=args.REPORT_TO,  # enable logging to W&B
            run_name=args.RUN_NAME,  # name of the W&B run (optional)
            load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
            logging_strategy="steps", # new
            logging_steps=args.LOGGING_STEPS,
            fp16=False,
            bf16=True, #True,
            adam_beta1= 0.9, #adjust adam
            adam_beta2= 0.95,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer,
        ckpt = args.RESUME_CKPT,
        data_path=args.DATA_PATH
    )
    elif args.CONTRASTIVE == 1:
        trainer = llama2_trainer_contrastive_comp(
            model=model,
            train_dataset=data['train'],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=args.MICRO_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                warmup_ratio=0.05,
                num_train_epochs=args.EPOCHS,
                learning_rate=args.LEARNING_RATE,
                save_strategy="steps",
                save_steps=args.SAVE_STEPS,
                eval_steps=args.EVAL_STEPS,
                output_dir=args.OUTPUT_DIR,
                overwrite_output_dir=True,
                save_total_limit=args.SAVE_TOTAL_LIMIT,
                evaluation_strategy=args.EVAL_STRATEGY,
                report_to=args.REPORT_TO,  # enable logging to W&B
                run_name=args.RUN_NAME,  # name of the W&B run (optional)
                load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
                logging_strategy="steps", # new
                logging_steps=args.LOGGING_STEPS,
                fp16=False,
                bf16=True, #True,
                adam_beta1= 0.9, #adjust adam
                adam_beta2= 0.95,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            tokenizer=tokenizer,
            ckpt = args.RESUME_CKPT,
            data_path=args.DATA_PATH,
            weight=args.CONTRASTIVE_WEIGHT
        )
    else:
        raise ValueError("Invalid contrastive type.")
    return trainer

def get_dataset(data_type, data_path):
    if data_type == "json":
        dataset = load_dataset("json", data_files=data_path)
    elif data_type == "txt":
        dataset = load_dataset("text", data_files=data_path)

    return dataset