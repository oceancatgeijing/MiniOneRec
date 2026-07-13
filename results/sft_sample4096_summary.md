# SFT sample4096 Summary

- script: scripts/sft_office_pro6000_sample4096.sh
- entry: sft_pro6000.py
- dataset: Office_Products
- base model: /root/autodl-tmp/models/Qwen2.5-1.5B
- sample: 4096
- train rows: 12288
- eval rows: 4096
- batch_size: 64
- micro_batch_size: 4
- steps: 192
- train_runtime: 302.2226
- train_loss: 2.224016277740399
- eval_loss: 2.9238033294677734
- observed peak GPU memory: about 19.4 GiB
- output_dir: output_dir/sft_office_pro6000_sample4096
- final_checkpoint: output_dir/sft_office_pro6000_sample4096/final_checkpoint/model.safetensors
- output size: 15G
- optimizer.pt: checkpoint-192/optimizer.pt, about 5.8G
- disk: /root/autodl-tmp 350G total, 54G used, 297G available

## Conclusion

sample4096 SFT 成功验证了单卡 RTX PRO 6000 上 MiniOneRec/Qwen2.5-1.5B 的中等规模 SFT 训练稳定性；相比 smoke_v2，样本扩大 4 倍、micro_batch_size 从 2 提升到 4，loss 正常下降，checkpoint/final_checkpoint 正常保存。
