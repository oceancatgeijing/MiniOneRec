#!/usr/bin/env python3
"""
RQ-VAE Training Metrics Comparator

Generates before/after comparison of EMA + Codebook Reset optimization.
Use after running two RQ-VAE training sessions:
  1. Baseline:  python rqvae.py --ema_decay 0.0 ...  (传统梯度更新)
  2. Optimized: python rqvae.py --ema_decay 0.99 --dead_threshold 2.0 ...

Usage:
  python rq/compare_metrics.py \
      --baseline output_dir/baseline/training_metrics.json \
      --optimized output_dir/optimized/training_metrics.json \
      --output comparison_report.md
"""

import json
import argparse
from pathlib import Path


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def last_n_avg(records, key, n=5):
    """Average of last N records for a scalar key."""
    vals = [r[key] for r in records[-n:] if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def compute_improvement(baseline_val, optimized_val):
    """Compute absolute and relative improvement."""
    abs_change = optimized_val - baseline_val
    rel_change = (abs_change / baseline_val * 100) if baseline_val > 0 else 0.0
    return abs_change, rel_change


def compare_metrics(baseline_path, optimized_path, output_path, window=5):
    """Generate comparison report."""
    baseline = load_metrics(baseline_path)
    optimized = load_metrics(optimized_path)

    # Use last N records for stable comparison
    b = {k: last_n_avg(baseline, k, window) for k in ['collision_rate', 'train_loss', 'train_recon_loss']}
    o = {k: last_n_avg(optimized, k, window) for k in ['collision_rate', 'train_loss', 'train_recon_loss']}

    # Codebook metrics from last record
    b_cb = baseline[-1].get('codebook', {}) if baseline else {}
    o_cb = optimized[-1].get('codebook', {}) if optimized else {}

    # ---- Build Report ----
    lines = []
    lines.append("# RQ-VAE EMA + Codebook Reset 优化效果对比")
    lines.append("")
    lines.append(f"**对比窗口**: 最后 `{window}` 个 eval step 的均值")
    lines.append(f"**Baseline**: `{baseline_path}` ({len(baseline)} records)")
    lines.append(f"**Optimized**: `{optimized_path}` ({len(optimized)} records)")
    lines.append("")

    # ---- Core metrics table ----
    lines.append("## 核心指标")
    lines.append("")
    lines.append("| 指标 | Baseline | Optimized | 绝对变化 | 相对变化 |")
    lines.append("|------|----------|-----------|----------|----------|")

    # Collision rate (lower is better)
    abs_c, rel_c = compute_improvement(b['collision_rate'], o['collision_rate'])
    direction = "↓ 改善" if abs_c < 0 else "↑ 恶化"
    lines.append(
        f"| **SID 碰撞率** | {b['collision_rate']:.4f} | {o['collision_rate']:.4f} "
        f"| {abs_c:+.4f} ({abs(rel_c):.1f}%) | {direction} |"
    )

    # Avg perplexity (higher is better)
    b_ppl = b_cb.get('avg_perplexity', 0)
    o_ppl = o_cb.get('avg_perplexity', 0)
    abs_p, rel_p = compute_improvement(b_ppl, o_ppl)
    direction_p = "↑ 提升" if abs_p > 0 else "↓ 下降"
    lines.append(
        f"| **Codebook Perplexity** | {b_ppl:.1f} | {o_ppl:.1f} "
        f"| {abs_p:+.1f} ({abs(rel_p):.1f}%) | {direction_p} |"
    )

    # Dead code ratio (lower is better)
    b_dead = b_cb.get('dead_code_ratio', 0)
    o_dead = o_cb.get('dead_code_ratio', 0)
    abs_d, rel_d = compute_improvement(b_dead, o_dead)
    direction_d = "↓ 改善" if abs_d < 0 else "↑ 恶化"
    lines.append(
        f"| **死码占比** | {b_dead:.1%} | {o_dead:.1%} "
        f"| {abs_d:+.1%} ({abs(rel_d):.1f}%) | {direction_d} |"
    )

    # Dead code count
    b_dc = b_cb.get('total_dead_codes', 0)
    o_dc = o_cb.get('total_dead_codes', 0)
    abs_dc = o_dc - b_dc
    lines.append(f"| **死码数量** | {b_dc} | {o_dc} | {abs_dc:+d} | |")

    lines.append("")

    # ---- Per-layer breakdown ----
    if b_cb.get('per_layer') and o_cb.get('per_layer'):
        lines.append("## 逐层 Codebook 利用率")
        lines.append("")
        lines.append("| Layer | Baseline PPL | Optimized PPL | 绝对提升 | 利用率 (Opt) |")
        lines.append("|-------|-------------|--------------|----------|-------------|")
        for layer_key in sorted(b_cb['per_layer'].keys()):
            bl = b_cb['per_layer'][layer_key]
            ol = o_cb['per_layer'][layer_key]
            ppl_diff = ol['perplexity'] - bl['perplexity']
            util = ol['perplexity'] / ol['codebook_size'] * 100 if ol['codebook_size'] > 0 else 0
            lines.append(
                f"| {layer_key} | {bl['perplexity']:.1f} | {ol['perplexity']:.1f} "
                f"| {ppl_diff:+.1f} | {util:.1f}% |"
            )
        lines.append("")

    # ---- Resume-ready summary ----
    lines.append("## 简历用关键数据")
    lines.append("")

    utilization_before = b_ppl / (b_cb.get('per_layer', {}).get('layer_0', {}).get('codebook_size', 256)) * 100
    utilization_after = o_ppl / (o_cb.get('per_layer', {}).get('layer_0', {}).get('codebook_size', 256)) * 100
    util_improve = utilization_after - utilization_before

    collision_improve = abs(rel_c) if abs_c < 0 else 0
    dead_improve = abs(rel_d) if abs_d < 0 else 0

    lines.append(f"- Codebook 空间利用率绝对提升：**{util_improve:.1f} 个百分点**（{utilization_before:.1f}% → {utilization_after:.1f}%）")
    lines.append(f"- SID 碰撞率相对降低：**{collision_improve:.1f}%**（{b['collision_rate']:.4f} → {o['collision_rate']:.4f}）")
    lines.append(f"- 死码占比相对降低：**{dead_improve:.1f}%**（{b_dead:.1%} → {o_dead:.1%}）")
    lines.append(f"- 平均 Codebook Perplexity 提升：**{o_ppl:.1f}**（Baseline: {b_ppl:.1f}）")
    lines.append("")

    # ---- Write ----
    output = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(output)
    print(output)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compare RQ-VAE training metrics (baseline vs EMA-optimized)"
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline training_metrics.json")
    parser.add_argument("--optimized", required=True, help="Path to optimized training_metrics.json")
    parser.add_argument("--output", default="comparison_report.md", help="Output markdown report")
    parser.add_argument("--window", type=int, default=5, help="Number of last records to average")
    args = parser.parse_args()

    compare_metrics(args.baseline, args.optimized, args.output, args.window)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
