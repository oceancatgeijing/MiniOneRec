#!/usr/bin/env bash
set -euo pipefail

# Complete remote pipeline for the resume-template experiments.
# Excluded by design: EMA/dead-code-reset and collaborative-vector RQ-VAE runs.

: "${BASE_MODEL:?Set BASE_MODEL to the remote Qwen2.5-1.5B path}"
: "${RAW_DATA_DIR:?Set RAW_DATA_DIR to the directory containing item/inter files}"
: "${TEXT_EMB_PATH:?Set TEXT_EMB_PATH to the text embedding .npy file}"
: "${CATEGORY:=Office_Products}"
: "${OUTPUT_ROOT:=output_dir/template_full}"
: "${RESULT_ROOT:=results/template_full}"
: "${SID_WORK_ROOT:=${OUTPUT_ROOT}/sid_data}"
: "${NUM_PROCESSES:=1}"
: "${MAIN_PROCESS_PORT:=29503}"
: "${GPU_LIST:=0}"
: "${FORCE:=0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/zero2_opt.yaml}"
ITEM_JSON_SOURCE="${ITEM_JSON_SOURCE:-${RAW_DATA_DIR}/${CATEGORY}.item.json}"
PRIMARY_SID_METHOD="${PRIMARY_SID_METHOD:-constrained_kmeans}"

RQ_DEVICE="${RQ_DEVICE:-cuda:0}"
RQ_EPOCHS="${RQ_EPOCHS:-5000}"
RQ_BATCH_SIZE="${RQ_BATCH_SIZE:-2048}"
RQ_LR="${RQ_LR:-1e-3}"
KMEANS_K="${KMEANS_K:-256}"
KMEANS_LEVELS="${KMEANS_LEVELS:-3}"
KMEANS_MAX_ITER="${KMEANS_MAX_ITER:-100}"

SFT_EPOCHS="${SFT_EPOCHS:-1}"
RL_EPOCHS="${RL_EPOCHS:-2}"
RECENT_N="${RECENT_N:-3}"
REWARD_TYPES="${REWARD_TYPES:-rule ranking sid_semantic debiased pop_penalty partial_match combined}"
COLD_RL_MODES="${COLD_RL_MODES:-sid_title sid_title_recent}"
COLD_REWARD_TYPE="${COLD_REWARD_TYPE:-combined}"

RUN_SID="${RUN_SID:-1}"
RUN_SID_SFT="${RUN_SID_SFT:-1}"
RUN_SFT_TASK_ABLATION="${RUN_SFT_TASK_ABLATION:-1}"
RUN_COLD_START_SFT="${RUN_COLD_START_SFT:-1}"
RUN_REWARD_ABLATION="${RUN_REWARD_ABLATION:-1}"
RUN_COLD_START_RL="${RUN_COLD_START_RL:-1}"

mkdir -p "${OUTPUT_ROOT}" "${RESULT_ROOT}" "${SID_WORK_ROOT}"

monitored() {
    local tag="$1"
    shift
    "${PYTHON_BIN}" resource_monitor.py \
        --output "${RESULT_ROOT}/${tag}_resources.json" -- "$@"
}

require_file() {
    [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }
}

link_raw_files() {
    local method_dir="$1"
    mkdir -p "${method_dir}"
    ln -sfn "$(realpath "${TEXT_EMB_PATH}")" "${method_dir}/${CATEGORY}.emb-qwen-td.npy"
    ln -sfn "$(realpath "${ITEM_JSON_SOURCE}")" "${method_dir}/${CATEGORY}.item.json"
    for split in train valid test; do
        local source="${RAW_DATA_DIR}/${CATEGORY}.${split}.inter"
        require_file "${source}"
        ln -sfn "$(realpath "${source}")" "${method_dir}/${CATEGORY}.${split}.inter"
    done
}

convert_sid_dataset() {
    local method="$1"
    local method_dir="${SID_WORK_ROOT}/${method}"
    local converted_dir="${method_dir}/converted"
    local done_file="${converted_dir}/.complete"
    if [[ -f "${done_file}" && "${FORCE}" != "1" ]]; then
        return
    fi
    "${PYTHON_BIN}" convert_dataset.py \
        --data_dir "${method_dir}" \
        --dataset_name "${CATEGORY}" \
        --output_dir "${converted_dir}" \
        --category "${CATEGORY}" \
        --seed 42
    touch "${done_file}"
}

run_rqvae_sid() {
    local method_dir="${SID_WORK_ROOT}/rqvae"
    local index_json="${method_dir}/${CATEGORY}.index.json"
    local metrics_json="${RESULT_ROOT}/sid_rqvae_metrics.json"
    link_raw_files "${method_dir}"
    if [[ ! -f "${index_json}" || "${FORCE}" == "1" ]]; then
        local ckpt_root="${OUTPUT_ROOT}/sid_rqvae_ckpt"
        mkdir -p "${ckpt_root}"
        monitored sid_rqvae_train "${PYTHON_BIN}" rq/rqvae.py \
            --data_path "${TEXT_EMB_PATH}" \
            --ckpt_dir "${ckpt_root}" \
            --lr "${RQ_LR}" \
            --epochs "${RQ_EPOCHS}" \
            --batch_size "${RQ_BATCH_SIZE}" \
            --device "${RQ_DEVICE}" \
            --ema_decay 0.0
        local ckpt_path
        ckpt_path=$(find "${ckpt_root}" -name best_collision_model.pth -type f -print | sort | tail -1)
        [[ -n "${ckpt_path}" ]] || { echo "RQ-VAE checkpoint not found" >&2; exit 1; }
        monitored sid_rqvae_index "${PYTHON_BIN}" rq/generate_rqvae_indices.py \
            --data_path "${TEXT_EMB_PATH}" \
            --ckpt_path "${ckpt_path}" \
            --output_path "${index_json}" \
            --metrics_path "${metrics_json}" \
            --device "${RQ_DEVICE}"
    fi
    convert_sid_dataset rqvae
}

run_constrained_kmeans_sid() {
    local method_dir="${SID_WORK_ROOT}/constrained_kmeans"
    local index_json="${method_dir}/${CATEGORY}.index.json"
    local metrics_json="${RESULT_ROOT}/sid_constrained_kmeans_metrics.json"
    link_raw_files "${method_dir}"
    if [[ ! -f "${index_json}" || "${FORCE}" == "1" ]]; then
        monitored sid_constrained_kmeans "${PYTHON_BIN}" rq/rqkmeans_constrained.py \
            --root "${method_dir}" \
            --dataset "${CATEGORY}" \
            --k "${KMEANS_K}" \
            --l "${KMEANS_LEVELS}" \
            --max_iter "${KMEANS_MAX_ITER}" \
            --seed 42 \
            --metrics_path "${metrics_json}" \
            --verbose
    fi
    convert_sid_dataset constrained_kmeans
}

method_paths() {
    local method="$1"
    METHOD_DIR="${SID_WORK_ROOT}/${method}"
    DATA_DIR="${METHOD_DIR}/converted"
    INDEX_JSON="${METHOD_DIR}/${CATEGORY}.index.json"
    ITEM_JSON="${METHOD_DIR}/${CATEGORY}.item.json"
    INFO_FILE="${DATA_DIR}/info/${CATEGORY}_5_2016-10-2018-11.txt"
    TRAIN_FILE="${DATA_DIR}/train/${CATEGORY}_5_2016-10-2018-11.csv"
    VALID_FILE="${DATA_DIR}/valid/${CATEGORY}_5_2016-10-2018-11.csv"
    TEST_FILE="${DATA_DIR}/test/${CATEGORY}_5_2016-10-2018-11.csv"
    require_file "${INDEX_JSON}"
    require_file "${ITEM_JSON}"
    require_file "${INFO_FILE}"
    require_file "${TRAIN_FILE}"
    require_file "${VALID_FILE}"
    require_file "${TEST_FILE}"
}

run_sft() {
    local method="$1" tag="$2" mode="$3" feature="$4" fusion="$5"
    method_paths "${method}"
    local out_dir="${OUTPUT_ROOT}/${tag}"
    if [[ -f "${out_dir}/final_checkpoint/config.json" && "${FORCE}" != "1" ]]; then
        echo "Skip existing SFT checkpoint: ${out_dir}"
        return
    fi
    monitored "${tag}_train" "${ACCELERATE_BIN}" launch \
        --config_file "${DEEPSPEED_CONFIG}" \
        --num_processes "${NUM_PROCESSES}" \
        --main_process_port "$((MAIN_PROCESS_PORT + 1))" \
        sft.py \
        --base_model "${BASE_MODEL}" \
        --train_file "${TRAIN_FILE}" \
        --eval_file "${VALID_FILE}" \
        --output_dir "${out_dir}" \
        --category "${CATEGORY}" \
        --sid_index_path "${INDEX_JSON}" \
        --item_meta_path "${ITEM_JSON}" \
        --history_mode "${mode}" \
        --recent_n "${RECENT_N}" \
        --enable_alignment_tasks True \
        --enable_feature_alignment "${feature}" \
        --enable_history_fusion "${fusion}" \
        --num_epochs "${SFT_EPOCHS}" \
        --train_from_scratch False \
        --freeze_LLM False
}

run_rl() {
    local method="$1" mode="$2" reward="$3" model_path="$4" tag="$5"
    method_paths "${method}"
    local out_dir="${OUTPUT_ROOT}/${tag}"
    if [[ -f "${out_dir}/final_checkpoint/config.json" && "${FORCE}" != "1" ]]; then
        echo "Skip existing RL checkpoint: ${out_dir}"
        return
    fi
    monitored "${tag}_train" "${ACCELERATE_BIN}" launch \
        --config_file "${DEEPSPEED_CONFIG}" \
        --num_processes "${NUM_PROCESSES}" \
        --main_process_port "${MAIN_PROCESS_PORT}" \
        rl.py \
        --model_path "${model_path}" \
        --train_file "${TRAIN_FILE}" \
        --eval_file "${VALID_FILE}" \
        --info_file "${INFO_FILE}" \
        --category "${CATEGORY}" \
        --sid_index_path "${INDEX_JSON}" \
        --item_meta_path "${ITEM_JSON}" \
        --history_mode "${mode}" \
        --recent_n "${RECENT_N}" \
        --reward_type "${reward}" \
        --num_train_epochs "${RL_EPOCHS}" \
        --beam_search True \
        --test_during_training False \
        --sync_ref_model True \
        --output_dir "${out_dir}" \
        --wandb_run_name "${tag}"
}

run_eval() {
    local method="$1" mode="$2" checkpoint="$3" tag="$4"
    method_paths "${method}"
    local result_json="${RESULT_ROOT}/${tag}.json"
    local metrics_json="${RESULT_ROOT}/${tag}_metrics.json"
    if [[ -f "${metrics_json}" && "${FORCE}" != "1" ]]; then
        echo "Skip existing metrics: ${metrics_json}"
        return
    fi
    monitored "${tag}_eval" env \
        MODEL_PATH="${checkpoint}" TEST_FILE="${TEST_FILE}" INFO_FILE="${INFO_FILE}" \
        CATEGORY="${CATEGORY}" RESULT_JSON="${result_json}" GPU_LIST="${GPU_LIST}" \
        HISTORY_MODE="${mode}" RECENT_N="${RECENT_N}" PYTHON_BIN="${PYTHON_BIN}" \
        bash run_parallel_evaluation.sh
    "${PYTHON_BIN}" experiment_metrics.py \
        --result_json "${result_json}" \
        --info_file "${INFO_FILE}" \
        --train_csv "${TRAIN_FILE}" \
        --output_json "${metrics_json}" \
        --sparse_history_lt 5 \
        --cold_item_max_count 0
}

if [[ "${RUN_SID}" == "1" ]]; then
    run_rqvae_sid
    run_constrained_kmeans_sid
fi

if [[ "${RUN_SID_SFT}" == "1" ]]; then
    for method in rqvae constrained_kmeans; do
        tag="sid_compare_${method}"
        run_sft "${method}" "${tag}" sid False False
        run_eval "${method}" sid "${OUTPUT_ROOT}/${tag}/final_checkpoint" "${tag}"
    done
fi

if [[ "${RUN_SFT_TASK_ABLATION}" == "1" ]]; then
    run_sft "${PRIMARY_SID_METHOD}" sft_task_sequence_only sid False False
    run_sft "${PRIMARY_SID_METHOD}" sft_task_feature_alignment sid True False
    run_sft "${PRIMARY_SID_METHOD}" sft_task_history_fusion sid False True
    run_sft "${PRIMARY_SID_METHOD}" sft_task_all sid True True
    for tag in sft_task_sequence_only sft_task_feature_alignment sft_task_history_fusion sft_task_all; do
        run_eval "${PRIMARY_SID_METHOD}" sid "${OUTPUT_ROOT}/${tag}/final_checkpoint" "${tag}"
    done
fi

if [[ "${RUN_COLD_START_SFT}" == "1" ]]; then
    for mode in sid title sid_title sid_title_recent; do
        tag="sft_cold_${mode}"
        run_sft "${PRIMARY_SID_METHOD}" "${tag}" "${mode}" True True
        run_eval "${PRIMARY_SID_METHOD}" "${mode}" \
            "${OUTPUT_ROOT}/${tag}/final_checkpoint" "${tag}"
    done
fi

if [[ "${RUN_REWARD_ABLATION}" == "1" ]]; then
    reward_base="${REWARD_BASE_MODEL:-${OUTPUT_ROOT}/sft_task_all/final_checkpoint}"
    for reward in ${REWARD_TYPES}; do
        tag="rl_reward_${reward}"
        run_rl "${PRIMARY_SID_METHOD}" sid "${reward}" "${reward_base}" "${tag}"
        run_eval "${PRIMARY_SID_METHOD}" sid \
            "${OUTPUT_ROOT}/${tag}/final_checkpoint" "${tag}"
    done
fi

if [[ "${RUN_COLD_START_RL}" == "1" ]]; then
    for mode in ${COLD_RL_MODES}; do
        tag="rl_cold_${mode}_${COLD_REWARD_TYPE}"
        run_rl "${PRIMARY_SID_METHOD}" "${mode}" "${COLD_REWARD_TYPE}" \
            "${OUTPUT_ROOT}/sft_cold_${mode}/final_checkpoint" "${tag}"
        run_eval "${PRIMARY_SID_METHOD}" "${mode}" \
            "${OUTPUT_ROOT}/${tag}/final_checkpoint" "${tag}"
    done
fi

"${PYTHON_BIN}" summarize_experiments.py \
    --metrics_dir "${RESULT_ROOT}" \
    --output_csv "${RESULT_ROOT}/experiment_summary.csv" \
    --output_md "${RESULT_ROOT}/experiment_summary.md"

echo "Complete experiment pipeline finished. Results: ${RESULT_ROOT}"
