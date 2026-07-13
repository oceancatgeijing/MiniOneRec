# MiniOneRec Official Checkpoint Evaluation

## Environment

- Platform: AutoDL / SeetaCloud
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- PyTorch: cu128, sm_120 supported
- Project path: /root/autodl-tmp/projects/MiniOneRec

## Office_Products

- Model: ./Office_ckpt
- Test samples: 4866
- Prediction file: results/eval_office.json
- Metrics file: results/metrics_office.log
- CC: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.0945335 | 0.0945335 |
| 3 | 0.12597616 | 0.112811 |
| 5 | 0.13892314 | 0.11826087 |
| 10 | 0.15885738 | 0.12468961 |
| 20 | 0.18392931 | 0.13097781 |
| 50 | 0.23715577 | 0.14141109 |

## Industrial_and_Scientific

- Model: ./Industrial_ckpt
- Test samples: 4533
- Prediction file: results/eval_industrial.json
- Metrics file: results/metrics_industrial.log
- CC: 0

| K | HR@K | NDCG@K |
|---:|---:|---:|
| 1 | 0.08515332 | 0.08515332 |
| 3 | 0.11294948 | 0.10101549 |
| 5 | 0.13236267 | 0.10905726 |
| 10 | 0.15729098 | 0.11699515 |
| 20 | 0.19258769 | 0.12591518 |
| 50 | 0.24442974 | 0.13612897 |

## Notes

- Evaluation uses evaluate.py to generate prediction JSON.
- Metrics are computed by calc.py.
- Short filename symlinks were created under ./data/Amazon to match script expectations.
