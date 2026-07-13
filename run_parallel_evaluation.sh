#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${TEST_FILE:?Set TEST_FILE}"
: "${INFO_FILE:?Set INFO_FILE}"
: "${CATEGORY:?Set CATEGORY}"
: "${RESULT_JSON:?Set RESULT_JSON}"
: "${GPU_LIST:=0}"
: "${HISTORY_MODE:=sid}"
: "${RECENT_N:=3}"
: "${NUM_BEAMS:=50}"
: "${BATCH_SIZE:=8}"

PYTHON_BIN="${PYTHON_BIN:-python}"
work_dir="${EVAL_WORK_DIR:-$(dirname "${RESULT_JSON}")/.eval_$(basename "${RESULT_JSON}" .json)}"
mkdir -p "${work_dir}" "$(dirname "${RESULT_JSON}")"

IFS=',' read -r -a gpus <<< "${GPU_LIST}"
"${PYTHON_BIN}" split.py \
    --input_path "${TEST_FILE}" \
    --output_path "${work_dir}" \
    --cuda_list "${GPU_LIST}"

pids=()
for gpu in "${gpus[@]}"; do
    shard="${work_dir}/${gpu}.csv"
    if [[ ! -f "${shard}" ]]; then
        echo "Missing evaluation shard: ${shard}" >&2
        exit 1
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" evaluate.py \
        --base_model "${MODEL_PATH}" \
        --info_file "${INFO_FILE}" \
        --category "${CATEGORY}" \
        --test_data_path "${shard}" \
        --result_json_data "${work_dir}/${gpu}.json" \
        --history_mode "${HISTORY_MODE}" \
        --recent_n "${RECENT_N}" \
        --batch_size "${BATCH_SIZE}" \
        --num_beams "${NUM_BEAMS}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if [[ "${status}" != "0" ]]; then
    echo "At least one evaluation worker failed" >&2
    exit "${status}"
fi

"${PYTHON_BIN}" merge.py \
    --input_path "${work_dir}" \
    --output_path "${RESULT_JSON}" \
    --cuda_list "${GPU_LIST}"

echo "Merged parallel evaluation output: ${RESULT_JSON}"
