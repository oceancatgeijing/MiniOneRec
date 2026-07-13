# Office full_1epoch Evaluation Summary

- checkpoint: output_dir/sft_office_pro6000_full_1epoch/final_checkpoint
- eval json: results/eval_office_full_1epoch.json
- eval log: results/eval_office_full_1epoch.log
- metrics log: results/metrics_office_full_1epoch.log
- test samples: 4866
- CC: 0

## Self-trained full_1epoch Metrics

- NDCG: [0.06699548, 0.08776832, 0.09432966, 0.10353097, 0.11246303, 0.12418309]
- HR: [0.06699548, 0.10213728, 0.11816687, 0.14673243, 0.18228524, 0.24167694]

## Official Office Checkpoint Reproduction Metrics

- NDCG: [0.0945335, 0.112811, 0.11826087, 0.12468961, 0.13097781, 0.14141109]
- HR: [0.0945335, 0.12597616, 0.13892314, 0.15885738, 0.18392931, 0.23715577]

## Conclusion

自训 full_1epoch checkpoint 在 HR@50 上达到 0.2417，略高于官方 checkpoint 复现值 0.2372，说明模型具备较好的候选召回能力；但 HR@1/3/5/10 和 NDCG@K 均低于官方 checkpoint，说明 top 排序质量仍有差距。该差距可能来自训练 epoch 数不足、官方训练包含更充分的 SID-text alignment/SFT/RL 阶段，或 checkpoint 训练策略与本次单卡复现不同。
