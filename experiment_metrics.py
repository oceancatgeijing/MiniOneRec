#!/usr/bin/env python3
"""Unified accuracy, long-tail, and cold-start evaluation for MiniOneRec."""

import argparse
import ast
import json
import math
import os
from collections import Counter

import numpy as np
import pandas as pd

from debiased_rewards import normalize_sid
from diversity_metrics import (
    compute_catalog_coverage,
    compute_novelty,
    compute_tail_coverage,
)


DEFAULT_TOPK = [1, 3, 5, 10, 20, 50]


def _history_length(sample):
    if "history_length" in sample:
        return int(sample["history_length"])
    for key in ("history_item_sid", "history"):
        if key not in sample:
            continue
        value = sample[key]
        if isinstance(value, list):
            return len(value)
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, list):
                return len(parsed)
        except (SyntaxError, ValueError):
            pass
    text = str(sample.get("input", ""))
    return text.count("<a_")


def load_catalog(info_file):
    catalog = set()
    with open(info_file, encoding="utf-8") as handle:
        for line in handle:
            sid = normalize_sid(line.split("\t", 1)[0])
            if sid:
                catalog.add(sid)
    return catalog


def load_popularity(train_csv):
    frame = pd.read_csv(train_csv, usecols=["item_sid"])
    counts = Counter(normalize_sid(value) for value in frame["item_sid"].dropna())
    return dict(counts), sum(counts.values())


def load_results(result_json):
    with open(result_json, encoding="utf-8") as handle:
        raw = json.load(handle)
    samples = []
    for item in raw:
        target = item.get("output", "")
        if isinstance(target, list):
            target = target[0] if target else ""
        predictions = item.get("predict", item.get("predictions", []))
        samples.append({
            "target": normalize_sid(target),
            "predictions": [normalize_sid(value) for value in predictions],
            "history_length": _history_length(item),
        })
    return samples


def compute_ranking_metrics(samples, topk):
    metrics = {"sample_count": len(samples), "hr": {}, "ndcg": {}}
    for k in topk:
        hits = []
        ndcgs = []
        for sample in samples:
            predictions = sample["predictions"][:k]
            try:
                rank = predictions.index(sample["target"])
            except ValueError:
                rank = None
            hits.append(float(rank is not None))
            ndcgs.append(0.0 if rank is None else 1.0 / math.log2(rank + 2))
        metrics["hr"][str(k)] = float(np.mean(hits)) if hits else 0.0
        metrics["ndcg"][str(k)] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return metrics


def compute_bucket(samples, catalog, item2pop, total, topk):
    metrics = compute_ranking_metrics(samples, topk)
    predictions = [sample["predictions"] for sample in samples]
    metrics["catalog_coverage"] = {
        str(k): value
        for k, value in compute_catalog_coverage(predictions, catalog, topk).items()
    }
    metrics["novelty"] = {
        str(k): value
        for k, value in compute_novelty(predictions, item2pop, total, topk).items()
    }
    metrics["tail_coverage"] = {
        str(k): value
        for k, value in compute_tail_coverage(predictions, item2pop, topk).items()
    }
    return metrics


def evaluate_results(
    result_json,
    info_file,
    train_csv,
    output_json=None,
    topk=None,
    sparse_history_lt=5,
    cold_item_max_count=0,
):
    topk = topk or DEFAULT_TOPK
    samples = load_results(result_json)
    catalog = load_catalog(info_file)
    item2pop, total = load_popularity(train_csv)

    buckets = {
        "all": samples,
        f"sparse_history_lt_{sparse_history_lt}": [
            sample for sample in samples
            if sample["history_length"] < sparse_history_lt
        ],
        f"cold_item_le_{cold_item_max_count}": [
            sample for sample in samples
            if item2pop.get(sample["target"], 0) <= cold_item_max_count
        ],
    }

    sorted_counts = sorted(item2pop.values())
    tail_cutoff = max(1, int(math.ceil(len(sorted_counts) * 0.8)))
    tail_threshold = sorted_counts[tail_cutoff - 1]
    buckets[f"long_tail_item_le_{tail_threshold}"] = [
        sample for sample in samples
        if item2pop.get(sample["target"], 0) <= tail_threshold
    ]

    report = {
        "result_json": result_json,
        "train_csv": train_csv,
        "catalog_size": len(catalog),
        "topk": topk,
        "buckets": {
            name: compute_bucket(values, catalog, item2pop, total, topk)
            for name, values in buckets.items()
        },
    }

    if output_json is None:
        stem, _ = os.path.splitext(result_json)
        output_json = stem + "_metrics.json"
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    print(f"Wrote metrics to {output_json}")
    for name, metrics in report["buckets"].items():
        print(
            f"{name}: n={metrics['sample_count']} "
            f"HR@10={metrics['hr'].get('10', 0):.4f} "
            f"NDCG@10={metrics['ndcg'].get('10', 0):.4f} "
            f"Coverage@10={metrics['catalog_coverage'].get('10', 0):.4f} "
            f"Novelty@10={metrics['novelty'].get('10', 0):.4f}"
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--info_file", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output_json")
    parser.add_argument("--topk", nargs="+", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--sparse_history_lt", type=int, default=5)
    parser.add_argument("--cold_item_max_count", type=int, default=0)
    args = parser.parse_args()
    evaluate_results(**vars(args))


if __name__ == "__main__":
    main()
