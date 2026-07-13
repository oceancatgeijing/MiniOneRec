# SFT Office full_1epoch Summary

- script: scripts/sft_office_pro6000_full_1epoch.sh
- entry: sft_pro6000.py
- dataset: Office_Products
- base model: /root/autodl-tmp/models/Qwen2.5-1.5B
- sample: -1 / full data
- train rows: 84735
- eval rows: 4866
- batch_size: 64
- micro_batch_size: 4
- num_epochs: 1
- steps: 1324
- train_runtime: 1912.106s
- train_loss: 0.9007771231931863
- eval_loss: 1.4612122774124146
- observed peak GPU memory: about 20.3 GiB
- output_dir: output_dir/sft_office_pro6000_full_1epoch
- final_checkpoint: output_dir/sft_office_pro6000_full_1epoch/final_checkpoint/model.safetensors
- output size: 15G
- optimizer.pt: checkpoint-1324/optimizer.pt, about 5.8G
- disk after training: /root/autodl-tmp has about 312G available

## Conclusion

full_1epoch SFT 成功完成了 Office_Products 全量数据上的 MiniOneRec/Qwen2.5-1.5B 单卡训练复现。训练共 1324 steps，loss 正常下降，epoch 末尾 eval_loss=1.4612，final_checkpoint 正常保存，峰值显存约 20.3GiB，说明 RTX PRO 6000 上该配置显存充足、训练链路稳定。
