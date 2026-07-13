#!/usr/bin/env python3
"""Collect experiment metric JSON files into resume-ready comparison tables."""

import argparse
import csv
import glob
import json
import os


def flatten_metrics(path):
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    experiment = os.path.basename(path).removesuffix("_metrics.json")
    if "buckets" not in report and "levels" in report:
        row = {
            "experiment": experiment,
            "bucket": "sid_construction",
            "sample_count": report.get("item_count", 0),
            "collision_rate": report.get("collision_rate", ""),
            "reconstruction_mse": report.get("reconstruction_mse", ""),
        }
        for level in report.get("levels", []):
            index = int(level.get("level", 0))
            row[f"codebook_utilization_l{index}"] = level.get("utilization", "")
            row[f"used_codes_l{index}"] = level.get("used_codes", "")
        return [row]
    rows = []
    for bucket, metrics in report.get("buckets", {}).items():
        row = {
            "experiment": experiment,
            "bucket": bucket,
            "sample_count": metrics.get("sample_count", 0),
        }
        for family in ("hr", "ndcg", "catalog_coverage", "novelty", "tail_coverage"):
            for k, value in metrics.get(family, {}).items():
                row[f"{family}@{k}"] = value
        rows.append(row)
    return rows


def flatten_resources(path):
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    return {
        "experiment": os.path.basename(path).removesuffix("_resources.json"),
        "bucket": "resources",
        "sample_count": "",
        "elapsed_seconds": report.get("elapsed_seconds", ""),
        "max_peak_gpu_memory_mib": report.get("max_peak_gpu_memory_mib", ""),
        "return_code": report.get("return_code", ""),
    }


def summarize(metrics_dir, output_csv, output_md):
    rows = []
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*_metrics.json"))):
        rows.extend(flatten_metrics(path))
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*_resources.json"))):
        rows.append(flatten_resources(path))
    if not rows:
        raise FileNotFoundError(f"No *_metrics.json files found in {metrics_dir}")

    preferred = [
        "experiment", "bucket", "sample_count", "hr@10", "ndcg@10",
        "catalog_coverage@10", "novelty@10", "tail_coverage@10",
        "collision_rate", "reconstruction_mse", "codebook_utilization_l0",
        "codebook_utilization_l1", "codebook_utilization_l2",
        "elapsed_seconds", "max_peak_gpu_memory_mib", "return_code",
    ]
    all_fields = set().union(*(row.keys() for row in rows))
    fields = preferred + sorted(all_fields - set(preferred))

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    display_fields = preferred
    with open(output_md, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(display_fields) + " |\n")
        handle.write("|" + "|".join(["---"] * len(display_fields)) + "|\n")
        for row in rows:
            values = []
            for field in display_fields:
                value = row.get(field, "")
                values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
            handle.write("| " + " | ".join(values) + " |\n")
    print(f"Wrote {len(rows)} rows to {output_csv} and {output_md}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    summarize(**vars(args))


if __name__ == "__main__":
    main()
