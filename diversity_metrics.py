#!/usr/bin/env python3
"""
Diversity & Coverage Metrics for Generative Recommendation
============================================================

Computes catalog coverage, novelty, and diversity metrics for evaluating
the long-tail performance of generative recommendation models.

Usage:
    python diversity_metrics.py \
        --pred_json results/eval_office.json \
        --info_file data/Amazon/info/Office_Products.txt \
        --topk 10 20 50

Or programmatically:
    from diversity_metrics import compute_all_metrics
    metrics = compute_all_metrics(predictions, item_sids, topk=[10, 20, 50])
"""

import argparse
import json
import math
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_catalog_coverage(
    predictions: List[List[str]],
    catalog_sids: Set[str],
    topk: List[int] = [10, 20, 50],
) -> Dict[int, float]:
    """
    Catalog Coverage: fraction of the full item catalog that appears in
    at least one user's top-K recommendations.

    Higher = better long-tail coverage.

    Args:
        predictions: List of per-user recommendation lists (each is top-K SIDs).
        catalog_sids: Set of all valid SIDs in the catalog.
        topk: List of K values to compute coverage at.

    Returns:
        Dict mapping K to coverage ratio.
    """
    coverage = {}
    for k in topk:
        recommended_sids = set()
        for user_recs in predictions:
            recommended_sids.update(user_recs[:k])
        covered = len(recommended_sids & catalog_sids)
        total = len(catalog_sids)
        coverage[k] = covered / total if total > 0 else 0.0
    return coverage


def compute_novelty(
    predictions: List[List[str]],
    item2pop: Dict[str, int],
    total_interactions: int,
    topk: List[int] = [10, 20, 50],
) -> Dict[int, float]:
    """
    Novelty: average negative log popularity of recommended items.
    Less popular items score higher novelty.

    Novelty@K = -1/|U| * Σ_u Σ_{i in topK(u)} log2(pop_i / total_interactions)

    Higher = more novel (less popular) recommendations.

    Args:
        predictions: Per-user recommendation lists.
        item2pop: Dict mapping SID string to interaction count.
        total_interactions: Total number of interactions in training data.
        topk: List of K values.

    Returns:
        Dict mapping K to novelty score.
    """
    novelty = {}
    for k in topk:
        scores = []
        for user_recs in predictions:
            user_novelty = 0.0
            count = 0
            for sid in user_recs[:k]:
                pop = item2pop.get(sid, 1)
                # -log2(pop / total) -> lower pop = higher novelty
                user_novelty += -math.log2(max(pop, 1) / total_interactions)
                count += 1
            if count > 0:
                scores.append(user_novelty / count)
        novelty[k] = float(np.mean(scores)) if scores else 0.0
    return novelty


def compute_intra_list_diversity(
    predictions: List[List[str]],
    sid_embeddings: Dict[str, np.ndarray],
    topk: List[int] = [10, 20, 50],
) -> Dict[int, float]:
    """
    Intra-List Diversity (ILD): average pairwise dissimilarity within each
    user's recommendation list. Uses cosine distance between SID embeddings.

    ILD@K = 1/|U| * Σ_u (1 - mean cosine similarity within user's topK)

    Higher = more diverse recommendations per user.

    Args:
        predictions: Per-user recommendation lists.
        sid_embeddings: Dict mapping SID to embedding vector.
        topk: List of K values.

    Returns:
        Dict mapping K to ILD score.
    """
    ild = {}
    for k in topk:
        scores = []
        for user_recs in predictions:
            recs = user_recs[:k]
            if len(recs) < 2:
                continue
            sims = []
            valid = 0
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    vi = sid_embeddings.get(recs[i])
                    vj = sid_embeddings.get(recs[j])
                    if vi is not None and vj is not None:
                        sim = np.dot(vi, vj) / (
                            np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8
                        )
                        sims.append(sim)
                        valid += 1
            if valid > 0:
                scores.append(1.0 - np.mean(sims))
        ild[k] = float(np.mean(scores)) if scores else 0.0
    return ild


def compute_tail_coverage(
    predictions: List[List[str]],
    item2pop: Dict[str, int],
    topk: List[int] = [10, 20, 50],
    tail_ratio: float = 0.8,
) -> Dict[int, float]:
    """
    Tail Coverage: fraction of long-tail items (bottom tail_ratio by popularity)
    that appear in recommendations.

    Tail items are those with popularity below the (1-tail_ratio) percentile.

    Args:
        predictions: Per-user recommendation lists.
        item2pop: Dict mapping SID to interaction count.
        topk: List of K values.
        tail_ratio: Fraction of items considered "tail" (default 0.8 = bottom 80%).

    Returns:
        Dict mapping K to tail coverage ratio.
    """
    # Identify tail items
    pops = sorted(item2pop.values())
    threshold_idx = int(len(pops) * (1 - tail_ratio))
    threshold = pops[threshold_idx] if threshold_idx < len(pops) else pops[-1]
    tail_sids = {sid for sid, pop in item2pop.items() if pop <= threshold}
    print(f"Tail items (pop <= {threshold}): {len(tail_sids)} / {len(item2pop)}")

    coverage = {}
    for k in topk:
        recommended = set()
        for user_recs in predictions:
            recommended.update(user_recs[:k])
        tail_recs = recommended & tail_sids
        coverage[k] = len(tail_recs) / len(tail_sids) if tail_sids else 0.0
    return coverage


# ---------------------------------------------------------------------------
# All-in-One
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: List[List[str]],
    catalog_sids: Set[str],
    item2pop: Dict[str, int],
    total_interactions: int,
    sid_embeddings: Dict[str, np.ndarray] = None,
    topk: List[int] = [10, 20, 50],
) -> Dict:
    """Compute all diversity metrics in one call."""
    metrics = {"topk": topk}

    metrics["catalog_coverage"] = compute_catalog_coverage(
        predictions, catalog_sids, topk
    )
    metrics["novelty"] = compute_novelty(
        predictions, item2pop, total_interactions, topk
    )
    metrics["tail_coverage"] = compute_tail_coverage(
        predictions, item2pop, topk
    )

    if sid_embeddings is not None:
        metrics["intra_list_diversity"] = compute_intra_list_diversity(
            predictions, sid_embeddings, topk
        )

    return metrics


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------

def load_predictions(json_path: str, beam_width: int = 50) -> List[List[str]]:
    """
    Load predictions from evaluate.py output JSON.

    The JSON has format:
        [{"input": ..., "output": ..., "predict": [...]}, ...]

    Returns:
        List of per-user recommendation lists (top-K SID strings).
    """
    with open(json_path) as f:
        data = json.load(f)

    predictions = []
    for item in data:
        preds = item.get("predict", [])
        if not preds:
            preds = item.get("predictions", [])
        predictions.append(preds[:beam_width])
    print(f"Loaded predictions: {len(predictions)} users, "
          f"beam_width={beam_width}")
    return predictions


def load_catalog(info_file: str) -> Set[str]:
    """Load all valid SIDs from info file."""
    sids = set()
    with open(info_file) as f:
        for line in f:
            sid = line.split("\t")[0].strip()
            if sid:
                sids.add(sid)
    print(f"Catalog: {len(sids)} unique SIDs")
    return sids


def build_popularity(train_csv: str) -> Tuple[Dict[str, int], int]:
    """Build SID popularity from training CSV."""
    import pandas as pd

    df = pd.read_csv(train_csv)
    pop = Counter()
    for sid in df["item_sid"]:
        pop[sid.strip()] += 1
    total = sum(pop.values())
    print(f"Popularity: {len(pop)} SIDs, {total} total interactions")
    return dict(pop), total


def build_sid_embeddings(index_json: str) -> Dict[str, np.ndarray]:
    """Build one-hot SID embeddings from index.json."""
    with open(index_json) as f:
        data = json.load(f)
    embeddings = {}
    for item_id, tokens in data.items():
        sid_str = "".join(tokens)
        vec = np.zeros(256 * 3, dtype=np.float32)
        for i, tok in enumerate(tokens):
            code = int(tok.split("_")[1].rstrip(">"))
            vec[i * 256 + code] = 1.0
        embeddings[sid_str] = vec / (np.linalg.norm(vec) + 1e-8)
    return embeddings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute diversity & coverage metrics for recommendations"
    )
    parser.add_argument("--pred_json", required=True,
                        help="Prediction JSON from evaluate.py")
    parser.add_argument("--info_file", required=True,
                        help="Catalog info file (SID per line)")
    parser.add_argument("--train_csv", default=None,
                        help="Training CSV for popularity stats")
    parser.add_argument("--index_json", default=None,
                        help="index.json for SID embeddings (ILD metric)")
    parser.add_argument("--topk", nargs="+", type=int, default=[10, 20, 50],
                        help="K values for metrics")
    parser.add_argument("--beam_width", type=int, default=50,
                        help="Number of beam candidates per user")
    args = parser.parse_args()

    predictions = load_predictions(args.pred_json, args.beam_width)
    catalog = load_catalog(args.info_file)

    item2pop, total = {}, 0
    if args.train_csv:
        item2pop, total = build_popularity(args.train_csv)

    sid_emb = None
    if args.index_json:
        sid_emb = build_sid_embeddings(args.index_json)

    metrics = compute_all_metrics(
        predictions, catalog, item2pop, total, sid_emb, args.topk
    )

    print(f"\n{'='*55}")
    print(f"  Diversity & Coverage Metrics")
    print(f"{'='*55}")
    for k in args.topk:
        cc = metrics["catalog_coverage"].get(k, 0)
        nv = metrics["novelty"].get(k, 0)
        tc = metrics["tail_coverage"].get(k, 0)
        print(f"  @{k:>2}: Coverage={cc:.1%}  Novelty={nv:.3f}  "
              f"TailCov={tc:.1%}")
    if "intra_list_diversity" in metrics:
        for k in args.topk:
            ild = metrics["intra_list_diversity"].get(k, 0)
            print(f"  @{k:>2}: ILD={ild:.3f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
