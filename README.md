# RECIPE-TKG: Reasoning-Centric Contrastive Fine-Tuning for Temporal Knowledge Graph Completion

This repository contains code, models, and data for experiments on temporal knowledge graph (TKG) completion using large language models (LLMs). We focus on enhancing reasoning via rule-based multi-hop history sampling and contrastive fine-tuning over datasets like ICEWS14, ICEWS18, GDELT, and YAGO.

---

## 🔧 Directory Structure

### 📁 `output/`
Stores the inference results for each experiment. Folder naming convention:


#### Data Sampling Shorthands:
- `tlr`: Temporal Logical Rule-based Sampling  
- `isi`: ICL-style Basic Sampling (as used in the ICL paper)  
- `rbmh`: Rule-Based Multi-Hop Sampling (our proposed method)

Each output subdirectory contains:
- `final.txt`: Cleaned inference results — final 10 predictions per test sample
- `raw.txt`: Unfiltered/raw output of the model
- `metric_results.txt`: Evaluation metrics (Hits@1, Hits@3, Hits@10)

#### Example Directories:
- `icews14_llama2_contrastive_rbmh_filtered`
- `icews14_llama2_semantic_diff_rbmh`
- `icews14_llama2chat_icl_tlr`
- `icews14_llama3_icl_rbmh`

---

### 📁 `model/`
Each subdirectory contains LoRA fine-tuned adapters. Directory naming follows the same format as `output/`.

Example:
```bash
model/
├── icews14_llama2_contrastive_rbmh/
│   ├── adapter_model.bin
│   ├── adapter_config.json
│   └
```

---

### 📁 `data/`
Organized into training and evaluation subsets:
```bash
data/
├── train/
│   ├── icews14/
│   ├── icews18/
│   ├── gdelt/
│   └── yago/
├── eval/
│   ├── icews14/
│   ├── icews18/
│   ├── gdelt/
│   └── yago/
```

Each dataset subfolder contains files sampled using `rbmh`, `tlr`, or `isi`.

---

## 🚀 Running the Code

### Fine-tuning
Run the following script to start fine-tuning:
```bash
bash train.sh
```

### Evaluation
Run the following script to start evaluation:
```bash
bash eval.sh
```

## 📄 Acknowledgments

This repository includes components adapted from [GenTKG](https://github.com/mayhugotong/GenTKG). Please refer to their [repository](https://github.com/mayhugotong/GenTKG) for the original code and license.
