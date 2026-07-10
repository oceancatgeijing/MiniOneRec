#!/bin/bash
# 自动训练 + 评估 + 计算 HR@K
# 用法: bash train_and_eval.sh <epoch_num>
# 示例: bash train_and_eval.sh 5

EPOCH=$1
PREV=$((EPOCH - 1))

if [ "$PREV" -eq 0 ]; then
    BASE_MODEL="/root/autodl-tmp/models/Qwen2.5-1.5B"
    FROM_SCRATCH="True"
else
    BASE_MODEL="output_dir/sft_office_collab_pre_${PREV}epoch/final_checkpoint"
    FROM_SCRATCH="False"
fi

OUT_DIR="output_dir/sft_office_collab_pre_${EPOCH}epoch"
EVAL_JSON="results/eval_office_collab_pre_epoch${EPOCH}.json"
INFO_FILE="data/Amazon_collab_pre_sk/info/Office_Products.txt"

echo "=== Epoch ${EPOCH}: Training ==="
/root/miniconda3/envs/minionerec/bin/python sft.py \
    --base_model ${BASE_MODEL} \
    --train_file data/Amazon_collab_pre_sk/train/Office_Products.csv \
    --eval_file data/Amazon_collab_pre_sk/valid/Office_Products.csv \
    --sid_index_path data/Amazon/index/Office_Products_collab_pre_sk.index.json \
    --item_meta_path data/Amazon/index/Office_Products.item.json \
    --category Office_Products --batch_size 64 --micro_batch_size 4 \
    --num_epochs 1 --output_dir ${OUT_DIR} \
    --train_from_scratch ${FROM_SCRATCH} --freeze_LLM False --seed 42 \
    --save_optimizer False 2>&1 | tee sft_epoch${EPOCH}.log

echo "=== Epoch ${EPOCH}: Evaluation ==="
/root/miniconda3/envs/minionerec/bin/python evaluate.py \
    --base_model ${OUT_DIR}/final_checkpoint \
    --info_file ${INFO_FILE} \
    --category Office_Products \
    --test_data_path data/Amazon_collab_pre_sk/test/Office_Products.csv \
    --num_beams 50 --K 3 --seed 42 \
    --result_json_data ${EVAL_JSON} 2>&1 | tee eval_epoch${EPOCH}.log

echo "=== Epoch ${EPOCH}: HR@K ==="
/root/miniconda3/envs/minionerec/bin/python calc.py ${EVAL_JSON} ${INFO_FILE}

echo "=== Epoch ${EPOCH} Complete ==="
