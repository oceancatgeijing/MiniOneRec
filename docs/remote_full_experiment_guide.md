# MiniOneRec Template Experiment Pipeline

This pipeline reproduces the experiment scope described by the resume template.
It intentionally excludes the following repository extensions:

- RQ-VAE with EMA and dead-code reset;
- RQ-VAE with collaborative-vector fusion.

The local workspace is used for code validation only. Training and generation
are expected to run on the remote GPU server.

## Default Experiment Matrix

### 1. Semantic-ID construction

| Tag | Method |
|---|---|
| `sid_rqvae` | Original text-only RQ-VAE, EMA explicitly disabled |
| `sid_constrained_kmeans` | Three-level constrained residual K-Means |

Both methods report per-level codebook utilization, collision rate, and, when
available, reconstruction MSE. Each SID mapping is converted into an isolated
train/validation/test dataset so the mappings cannot overwrite each other.

### 2. SID downstream comparison

| Tag | SID method | SFT tasks |
|---|---|---|
| `sid_compare_rqvae` | RQ-VAE | Sequence recommendation only |
| `sid_compare_constrained_kmeans` | Constrained K-Means | Sequence recommendation only |

These two runs use the same base model and SFT settings.

### 3. SFT task ablation

All runs use `PRIMARY_SID_METHOD`, which defaults to `constrained_kmeans`.

| Tag | Sequence | Feature alignment | History fusion |
|---|---:|---:|---:|
| `sft_task_sequence_only` | Yes | No | No |
| `sft_task_feature_alignment` | Yes | Yes | No |
| `sft_task_history_fusion` | Yes | No | Yes |
| `sft_task_all` | Yes | Yes | Yes |

### 4. Cold-start prompt ablation

| Tag | History representation |
|---|---|
| `sft_cold_sid` | Semantic IDs only |
| `sft_cold_title` | Product titles only |
| `sft_cold_sid_title` | SID and title pairs |
| `sft_cold_sid_title_recent` | SID/title pairs plus recent-title preference |

### 5. GRPO reward ablation

The default runner trains seven GRPO variants from `sft_task_all`:

`rule`, `ranking`, `sid_semantic`, `debiased`, `pop_penalty`,
`partial_match`, and `combined`.

The SFT checkpoint itself is the no-RL baseline. `partial_match` supplies
hierarchical SID rewards of 0.2/0.5/1.0. `combined` mixes the partial signal with
inverse-popularity and candidate-novelty rewards.

### 6. Cold-start plus GRPO

| Tag | Prompt | Reward |
|---|---|---|
| `rl_cold_sid_title_combined` | SID + title | Combined debiased reward |
| `rl_cold_sid_title_recent_combined` | SID + title + recent preference | Combined debiased reward |

The evaluator reports full-test, history-length `<5`, unseen-target, and bottom
80% long-tail buckets. Metrics include HR, NDCG, Catalog Coverage, Novelty, and
Tail Coverage.

The default matrix contains 19 downstream training runs: 2 SID comparison SFT,
4 task-ablation SFT, 4 cold-start SFT, 7 reward GRPO, and 2 cold-start GRPO. SID
construction adds one RQ-VAE training run and one constrained K-Means job.

## Remote Configuration

`RAW_DATA_DIR` must contain:

```text
Office_Products.item.json
Office_Products.train.inter
Office_Products.valid.inter
Office_Products.test.inter
```

Example:

```bash
export BASE_MODEL=/remote/models/Qwen2.5-1.5B
export RAW_DATA_DIR=/remote/data/Amazon18/Office_Products
export TEXT_EMB_PATH=/remote/data/Amazon18/Office_Products/Office_Products.emb-qwen-td.npy
export CATEGORY=Office_Products
export OUTPUT_ROOT=/remote/outputs/minionerec_template
export RESULT_ROOT=/remote/results/minionerec_template
export NUM_PROCESSES=8
export GPU_LIST=0,1,2,3,4,5,6,7

bash run_resume_experiments.sh
```

SFT and GRPO use `config/zero2_opt.yaml`, enabling ZeRO-2 and bf16. Evaluation
is split across all devices in `GPU_LIST` and merged before metrics are computed.
Every training and evaluation command is wrapped by `resource_monitor.py`, which
records elapsed seconds and per-GPU peak memory in `*_resources.json`. These
reports provide the evidence for runtime, multi-GPU inference, and peak-memory
claims without adding another full training matrix.

## Staged Execution

Every stage skips completed artifacts unless `FORCE=1` is set. Expensive stages
can therefore be submitted as separate remote jobs:

```bash
# SID construction and conversion only
RUN_SID=1 RUN_SID_SFT=0 RUN_SFT_TASK_ABLATION=0 \
RUN_COLD_START_SFT=0 RUN_REWARD_ABLATION=0 RUN_COLD_START_RL=0 \
  bash run_resume_experiments.sh

# SID downstream comparison and SFT experiments
RUN_SID=0 RUN_SID_SFT=1 RUN_SFT_TASK_ABLATION=1 \
RUN_COLD_START_SFT=1 RUN_REWARD_ABLATION=0 RUN_COLD_START_RL=0 \
  bash run_resume_experiments.sh

# GRPO experiments
RUN_SID=0 RUN_SID_SFT=0 RUN_SFT_TASK_ABLATION=0 \
RUN_COLD_START_SFT=0 RUN_REWARD_ABLATION=1 RUN_COLD_START_RL=1 \
  bash run_resume_experiments.sh
```

Final artifacts:

- raw beam prediction JSON files;
- per-experiment `_metrics.json` reports;
- `experiment_summary.csv`;
- `experiment_summary.md`.

Only compare percentage improvements when dataset split, beam width, seed, base
model, and training budget are identical.
