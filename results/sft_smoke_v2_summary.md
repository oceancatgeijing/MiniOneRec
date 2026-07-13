# SFT smoke_v2 Summary

- script: scripts/sft_office_pro6000_smoke_v2.sh
- entry: sft_pro6000.py
- dataset: Office_Products
- base model: /root/autodl-tmp/models/Qwen2.5-1.5B
- sample: 1024
- train rows: 3072
- eval rows: 1024
- batch_size: 64
- micro_batch_size: 2
- steps: 48
- train_runtime: 157.4712
- train_loss: 3.2232587983210883
- eval_loss: 3.2417330741882324
- observed peak GPU memory: about 17.5 GiB
- output_dir: output_dir/sft_office_pro6000_smoke_v2
- final_checkpoint: output_dir/sft_office_pro6000_smoke_v2/final_checkpoint/model.safetensors
- output size: 15G
- optimizer.pt: checkpoint-48/optimizer.pt, about 5.8G

## Conclusion

smoke_v2 SFT 成功验证了 sft_pro6000.py 的 epoch-level eval/save 策略有效，避免了原 sft.py 每约 5% steps 保存 checkpoint 导致的频繁 optimizer.pt 写入问题；该实验确认 MiniOneRec/Qwen2.5-1.5B 在单卡 RTX PRO 6000 上可稳定完成 Office_Products 小规模 SFT。
