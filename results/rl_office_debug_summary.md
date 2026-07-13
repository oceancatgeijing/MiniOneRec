# RL Office Debug Summary

- script: scripts/rl_office_pro6000_debug.sh
- entry: rl_pro6000_debug.py
- base checkpoint: output_dir/sft_office_pro6000_full_1epoch/final_checkpoint
- dataset: Office_Products
- train rows: 384
- eval rows: 128
- train steps: 384
- num_generations: 4
- reward_type: ranking
- beam_search: True
- train_runtime: 188.2877
- train_loss: 0.00021698789579479202
- eval_loss: 0.002977680414915085
- eval_reward: 0.01468641747487709
- eval_rewards/rule_reward: 0.041015625
- eval_rewards/ndcg_rule_reward: -0.026329208631068468
- eval_kl: 2.9866867065429688
- observed peak GPU memory: about 45.1 GiB
- output_dir: output_dir/rl_office_pro6000_debug
- checkpoint: output_dir/rl_office_pro6000_debug/checkpoint-384
- final_checkpoint: output_dir/rl_office_pro6000_debug/final_checkpoint
- output size: about 30G
- note: DeepSpeed cleanup warning appeared after training finished, but exit code was 0.

## Conclusion

GRPO debug 成功验证了 MiniOneRec 在单卡 RTX PRO 6000 上的 SFT → GRPO/RL 训练链路。虽然该实验仅使用 128 sample 的 debug 副本，结果不能代表最终推荐效果，但已经验证了 beam_search、num_generations=4、ranking reward、rule reward、KL 约束和 checkpoint 保存流程均可运行。后续可基于该链路扩大样本或进一步评估 RL checkpoint。
