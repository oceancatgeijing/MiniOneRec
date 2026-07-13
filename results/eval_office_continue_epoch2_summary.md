# Office_Products Continue Epoch2 SFT and Evaluation Summary

## Training Result

- checkpoint: `output_dir/sft_office_pro6000_continue_epoch2/final_checkpoint`
- train rows: 84735
- eval rows: 4866
- steps: 1324
- train_runtime: 1936.9656s
- train_loss: 0.5008011465706494
- eval_loss: 1.380003571510315
- observed peak GPU memory: about 20.3 GiB
- output size: 15G
- optimizer.pt: `checkpoint-1324/optimizer.pt`, about 5.8G

## Evaluation Result

- eval json: `results/eval_office_continue_epoch2.json`
- metrics log: `results/metrics_office_continue_epoch2.log`
- test samples: 4866
- CC: 0
- HR: [0.07727086, 0.10686395, 0.12227702, 0.13851212, 0.16420058, 0.21701603]
- NDCG: [0.07727086, 0.09491955, 0.10129643, 0.10650435, 0.1128923, 0.12339573]

## Full 1 Epoch Metrics

- HR: [0.06699548, 0.10213728, 0.11816687, 0.14673243, 0.18228524, 0.24167694]
- NDCG: [0.06699548, 0.08776832, 0.09432966, 0.10353097, 0.11246303, 0.12418309]

## Official Office Checkpoint Metrics

- HR: [0.0945335, 0.12597616, 0.13892314, 0.15885738, 0.18392931, 0.23715577]
- NDCG: [0.0945335, 0.112811, 0.11826087, 0.12468961, 0.13097781, 0.14141109]

## Conclusion

continue_epoch2 的 eval_loss 相比 full_1epoch 从 1.4612 降至 1.3800，说明 SFT 目标仍在优化；但推荐指标并非全面提升。HR@1 和 NDCG@1/3/5/10/20 有提升，说明头部排序有所改善；但 HR@10/20/50 与 NDCG@50 下降，说明更深候选召回能力变弱。该结果表明 SFT loss 下降不必然带来 HR/NDCG 全面提升，后续更适合尝试 GRPO/排序奖励或解码策略，而不是盲目继续增加 SFT epoch。
