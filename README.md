# Latent Geometric Denoising for Precise Knowledge Boundary Awareness

This repository contains the official code for the ACL 2026 paper:

> **Latent Geometric Denoising for Precise Knowledge Boundary Awareness**

---

## Overview

Large language models (LLMs) frequently hallucinate — confidently generating plausible but incorrect answers. A key obstacle is that models fail to distinguish what they truly know from what they do not. This paper proposes a **probe-guided curriculum SFT** framework that:

1. Trains a **linear probe** on the model's hidden states to identify the knowledge boundary — separating questions the model can answer correctly from those it cannot.
2. Uses the probe's **signed decision distance** to rank and select training samples, focusing SFT on the most informative boundary cases.
3. Fine-tunes the model with **boundary-aware abstention supervision**: known questions are trained to answer correctly; unknown questions are trained to abstain ("I don't know.").

The result is a model that accurately answers questions within its knowledge and reliably abstains on questions outside it, reducing hallucination without sacrificing helpfulness.

---

## Repository Structure

```
Anonymous_code/
├── training/
│   └── train.sh              # SFT training script (ms-swift, DeepSpeed ZeRO-3)
├── probe/
│   ├── get_hidden_TBG.py     # Extract hidden states (question only, TBG setting)
│   ├── get_hidden_SLT.py     # Extract hidden states (question + low-temp answer, SLT setting)
│   ├── train_probe.py        # Train per-layer logistic regression probes
│   ├── test_probe.py         # Evaluate trained probes; outputs per-sample distances
│   └── construct_dataset.py  # Build boundary-aware SFT dataset from probe distances
├── evaluation/
│   ├── test_qa.py            # Inference + evaluation on QA benchmarks (vLLM)
│   ├── llm_judge.py          # LLM-as-judge re-evaluation for semantic correctness
│   ├── compute_metric.py     # Compute F1-rel and precision/recall for ans/abstention
│   ├── alcuna.json           # ALCUNA test set (unanswerable questions)
│   ├── selfaware.json        # SelfAware test set
│   └── falseqa.json          # FalseQA test set
└── requirements.txt          # Python dependencies
```

---

## Method

### Step 1 — Hidden State Extraction

Extract per-layer hidden states of the last token for each training question.

**TBG** (question only):
```bash
python probe/get_hidden_TBG.py \
    --input_json  <path/to/train_data.json> \
    --model_name  <path/to/base_model> \
    --max_length  512 \
    --gpu         0
# Output: <input_json>_qhs.pkl
```

**SLT** (question + low-temperature model answer):
```bash
python probe/get_hidden_SLT.py \
    --input_json  <path/to/train_data.json> \
    --model_name  <path/to/base_model> \
    --max_length  512 \
    --gpu         0
# Output: <input_json>_qahs.pkl
```

Both scripts output a `.pkl` file where each sample contains:
- `question_hs`: `[num_layers, hidden_size]` hidden states
- `correct`: ground-truth label (1 = answerable, 0 = not)

### Step 2 — Probe Training

Train a logistic regression probe for every transformer layer:

```bash
python probe/train_probe.py \
    --data_path  <path/to/hidden_states.pkl> \
    --output_dir <path/to/probe_output/> \
    --max-iter   1000
```

Outputs per-layer probe models (`.pkl`) and metrics (`.json`).

### Step 3 — Probe Testing & Distance Extraction

Evaluate probes on a held-out set and extract per-sample signed distances to the decision boundary:

```bash
python probe/test_probe.py \
    --model_dir    <path/to/probe_output/> \
    --model_prefix <prefix_used_during_training> \
    --data_path    <path/to/test_hidden_states.pkl> \
    --output_dir   <path/to/test_results/> \
    --max_samples  10000
```

### Step 4 — Boundary-Aware Dataset Construction

Select the most informative samples (farthest from the boundary on each side) for SFT:

```bash
python probe/construct_dataset.py \
    --probe_results_path <path/to/test_results/layer_XX_results.json> \
    --origin_data_path   <path/to/train_data.json> \
    --output_path        <path/to/sft_dataset.json> \
    --threshold          50
```

`--threshold X` selects the top X% of samples farthest from the decision boundary on each side (negative = abstain, positive = answer). Default is `100` (keep all). The resulting dataset assigns `"I don't know."` to unknowable questions and the ground-truth answer to knowable ones.

### Step 5 — SFT Training

Fine-tune the base model with the constructed dataset using ms-swift:

```bash
# Edit training/train.sh to set model_path, input_data_path, output_dir, then:
bash training/train.sh
```

Key hyperparameters (see `train.sh`):
| Parameter | Value |
|---|---|
| Training type | Full fine-tuning |
| dtype | bfloat16 |
| Learning rate | 1e-5 |
| Batch size (per device) | 1 |
| Gradient accumulation | 4 |
| Max sequence length | 8192 |
| Warmup ratio | 0.05 |
| Parallelism | DeepSpeed ZeRO-3 |
| Attention | FlashAttention-2 |

---

## Evaluation

### QA Benchmarks

Evaluate on TriviaQA, NQ, SciQ, SimpleQA, ALCUNA, SelfAware, and FalseQA:

```bash
python evaluation/test_qa.py \
    --model_path <path/to/finetuned_model> \
    --output_dir <path/to/results/> \
    --gpu_id     0
```

Each sample is labeled with a **status**: `0` = abstained, `1` = correct, `2` = hallucination.

### LLM-as-Judge Re-evaluation

For semantic correctness beyond exact string match:

```bash
python evaluation/llm_judge.py \
    --model_path <path/to/judge_model> \
    --data_path  <path/to/results/model_name_dataset.json> \
    --gpu_id     0
```

### Metric Computation (F1-rel)

Compute precision/recall/F1 for both answering and abstention, plus the harmonic mean F1-rel:

```bash
python evaluation/compute_metric.py \
    --ref_path  <path/to/train_data.json> \
    --pred_path <path/to/results/model_name_dataset.json>
```

**Metrics:**
- `F1_ans` — F1 for correctly answered questions
- `F1_abs` — F1 for correctly abstained questions
- `F1_rel` — Harmonic mean of F1_ans and F1_abs (primary metric)

---

## Requirements

```bash
pip install -r requirements.txt
```

Requires Python 3.10/3.11 and CUDA 12.x. Key dependencies: `torch==2.8.0+cu126`, `transformers==4.57.3`, `vllm==0.11.0`, `ms_swift==3.11.0`, `deepspeed==0.17.6`, `flash_attn==2.8.3`.

---

## Datasets Used

| Dataset | Split | Purpose |
|---|---|---|
| TriviaQA | validation | In-domain evaluation |
| NQ (Natural Questions) | validation | Open-domain QA |
| SciQ | test | Science QA |
| SimpleQA | test | Factual QA |
| ALCUNA | test | Unanswerable (entity-level) |
| SelfAware | test | Unanswerable (self-knowledge) |
| FalseQA | test | False-premise questions |
