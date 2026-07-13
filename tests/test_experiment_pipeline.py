import json

import pandas as pd
import pytest

from data import build_recommendation_prompt
from debiased_rewards import (
    combined_debiased_reward,
    debiased_reward,
    normalize_sid,
    partial_match_reward,
)
from experiment_metrics import evaluate_results
from summarize_experiments import flatten_metrics, flatten_resources


def test_normalize_sid_extracts_model_output():
    assert normalize_sid('### Response:\n"<a_1><b_2><c_3>"') == "<a_1><b_2><c_3>"


def test_partial_match_reward_has_three_distinct_levels():
    target = ["<a_1><b_2><c_3>"] * 4
    completions = [
        "<a_1><b_2><c_3>",
        "<a_1><b_2><c_9>",
        "<a_1><b_8><c_9>",
        "<a_7><b_8><c_9>",
    ]
    assert partial_match_reward([], completions, target) == [1.0, 0.5, 0.2, 0.0]


def test_debiased_reward_upweights_tail_matches():
    completions = ["<a_1><b_1><c_1>", "<a_2><b_2><c_2>"]
    rewards = debiased_reward(
        [], completions, completions,
        item2pop={completions[0]: 100, completions[1]: 1},
        popularity_alpha=0.5,
        max_debiased_reward=20.0,
    )
    assert rewards[1] > rewards[0]


def test_debiased_reward_distinguishes_incorrect_candidates():
    rewards = debiased_reward(
        [], ["<a_1><b_1><c_1>", "<a_2><b_2><c_2>"],
        ["<a_9><b_9><c_9>"] * 2,
        item2pop={"<a_1><b_1><c_1>": 100, "<a_2><b_2><c_2>": 1},
        novelty_bonus=0.1,
    )
    assert rewards[1] > rewards[0]


def test_combined_reward_keeps_partial_signal():
    reward = combined_debiased_reward(
        [], ["<a_1><b_2><c_9>"], ["<a_1><b_2><c_3>"],
        item2pop={"<a_1><b_2><c_3>": 10},
    )
    assert reward == pytest.approx([0.26])


def test_history_prompt_modes():
    row = {
        "history_item_sid": "['<a_1><b_2><c_3>', '<a_4><b_5><c_6>']",
        "history_item_title": "['Paper', 'Ink']",
    }
    assert "semantic IDs:" in build_recommendation_prompt(row, "sid")
    assert "item titles:" in build_recommendation_prompt(row, "title")
    assert "title: \"Paper\"" in build_recommendation_prompt(row, "sid_title")
    assert "most recent preference signal" in build_recommendation_prompt(
        row, "sid_title_recent", recent_n=1
    )


def test_unified_metrics_and_sparse_bucket(tmp_path):
    train_csv = tmp_path / "train.csv"
    pd.DataFrame({
        "item_sid": ["<a_1><b_1><c_1>", "<a_1><b_1><c_1>", "<a_2><b_2><c_2>"]
    }).to_csv(train_csv, index=False)

    info_file = tmp_path / "info.txt"
    info_file.write_text(
        "<a_1><b_1><c_1>\tOne\t1\n<a_2><b_2><c_2>\tTwo\t2\n",
        encoding="utf-8",
    )
    result_json = tmp_path / "results.json"
    result_json.write_text(json.dumps([
        {
            "output": "<a_1><b_1><c_1>\n",
            "predict": ["<a_1><b_1><c_1>", "<a_2><b_2><c_2>"],
            "history_length": 2,
        },
        {
            "output": "<a_2><b_2><c_2>\n",
            "predict": ["<a_1><b_1><c_1>", "<a_2><b_2><c_2>"],
            "history_length": 8,
        },
    ]), encoding="utf-8")

    report = evaluate_results(
        str(result_json), str(info_file), str(train_csv),
        output_json=str(tmp_path / "metrics.json"), topk=[1, 2],
    )
    assert report["buckets"]["all"]["hr"]["2"] == 1.0
    assert report["buckets"]["sparse_history_lt_5"]["sample_count"] == 1


def test_summary_supports_sid_and_resource_reports(tmp_path):
    sid_path = tmp_path / "sid_rqvae_metrics.json"
    sid_path.write_text(json.dumps({
        "item_count": 10,
        "collision_rate": 0.1,
        "levels": [{"level": 0, "utilization": 0.5, "used_codes": 128}],
    }), encoding="utf-8")
    sid_rows = flatten_metrics(str(sid_path))
    assert sid_rows[0]["bucket"] == "sid_construction"
    assert sid_rows[0]["codebook_utilization_l0"] == 0.5

    resource_path = tmp_path / "run_resources.json"
    resource_path.write_text(json.dumps({
        "return_code": 0,
        "elapsed_seconds": 12.5,
        "max_peak_gpu_memory_mib": 4096,
    }), encoding="utf-8")
    resource = flatten_resources(str(resource_path))
    assert resource["elapsed_seconds"] == 12.5
    assert resource["max_peak_gpu_memory_mib"] == 4096
