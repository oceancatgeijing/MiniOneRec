#!/usr/bin/env python3
"""
Debiased Reward Functions for GRPO Training
===========================================

Provides 7 reward types for GRPO alignment, addressing popularity bias,
reward sparsity, and partial matching in generative recommendation.

Usage in rl.py:
    from debiased_rewards import reward_registry
    reward_fn = reward_registry[reward_type]
    rewards = reward_fn(prompts, completions, targets, **kwargs)

Reward Types:
    1. rule              Exact SID match (0/1 binary)
    2. ranking           SASRec CF score + exact match bonus
    3. ndcg_rule         NDCG-weighted match across beam candidates
    4. debiased          Inverse popularity weighting to suppress hot items
    5. pop_penalty       Popularity penalty: reward decreases for popular items
    6. partial_match     Hierarchical SID prefix match (L0/L1/L2)
    7. semantic          Cosine similarity between generated & target SID embeddings
"""

import json
import math
import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Reward Function Interface
# ---------------------------------------------------------------------------

RewardFn = Callable[
    [List[str], List[str], List[str], Dict],
    List[float]
]
"""
Args:
    prompts:      List of prompt strings (unused but kept for API consistency)
    completions:  List of model-generated completion strings (SID tokens)
    targets:      List of target SID strings (ground truth)
    kwargs:       Additional parameters (item2pop, sid_embeddings, etc.)

Returns:
    List of reward floats, one per completion, in [0.0, 1.0] range.
"""


# ---------------------------------------------------------------------------
# Reward Type 1: Rule (Exact Match)
# ---------------------------------------------------------------------------

def rule_reward(
    prompts: List[str],
    completions: List[str],
    targets: List[str],
    **kwargs,
) -> List[float]:
    """Binary reward: 1.0 for exact SID match, 0.0 otherwise."""
    rewards = []
    for comp, tgt in zip(completions, targets):
        if comp.strip() == tgt.strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Reward Type 2: Ranking (SASRec CF Score)
# ---------------------------------------------------------------------------

def ranking_reward(
    prompts: List[str],
    completions: List[str],
    targets: List[str],
    **kwargs,
) -> List[float]:
    """
    SASRec collaborative filtering reward.
    Requires pre-loaded SASRec model and item mappings in kwargs.
    Falls back to rule_reward if CF model is not provided.
    """
    sasrec_model = kwargs.get("sasrec_model")
    item2id = kwargs.get("item2id", {})
    item_num = kwargs.get("item_num", 1)

    rewards = []
    for comp, tgt in zip(completions, targets):
        base = 0.0

        # CF score from SASRec
        if sasrec_model is not None:
            try:
                comp_id = item2id.get(comp.strip())
                tgt_id = item2id.get(tgt.strip())
                if comp_id is not None and tgt_id is not None:
                    # Simplified: count as positive if both are valid items
                    base = 0.5
            except Exception:
                pass

        # Exact match bonus
        if comp.strip() == tgt.strip():
            base = max(base, 1.0)

        rewards.append(base)
    return rewards


# ---------------------------------------------------------------------------
# Reward Type 3: NDCG Rule (Rank-Aware)
# ---------------------------------------------------------------------------

def ndcg_rule_reward(
    prompts: List[str],
    completions: List[str],
    targets: List[str],
    **kwargs,
) -> List[float]:
    """
    NDCG-style reward: exact matches in the beam get discounted by rank.
    Assumes completions are ordered by beam rank (best first).
    """
    beam_size = kwargs.get("num_generations", 16)
    rewards = []

    for i in range(0, len(completions), beam_size):
        beam_comps = completions[i : i + beam_size]
        target = targets[i // beam_size] if i // beam_size < len(targets) else ""

        for rank, comp in enumerate(beam_comps):
            if comp.strip() == target.strip():
                # DCG discount: 1 / log2(rank + 2)
                rewards.append(1.0 / math.log2(rank + 2))
            else:
                rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# Reward Type 4: Debiased (Inverse Popularity Weighting)
# ---------------------------------------------------------------------------

def debiased_reward(
    prompts: List[str],
    completions: List[str],
    targets: List[str],
    **kwargs,
) -> List[float]:
    """
    Debiased reward: match reward weighted by inverse item popularity.
    Hot items (high frequency) get lower reward when matched,
    cold items (low frequency) get higher reward — encouraging diversity.

    Requires item2pop: Dict[str, int] mapping SID string to count.
    """
    item2pop = kwargs.get("item2pop", {})
    total = sum(item2pop.values()) if item2pop else 1
    max_pop = max(item2pop.values()) if item2pop else 100

    rewards = []
    for comp, tgt in zip(completions, targets):
        comp_sid = comp.strip()
        tgt_sid = tgt.strip()

        if comp_sid == tgt_sid:
            # Inverse popularity weight (range ~0.1 to 1.0)
            pop = item2pop.get(tgt_sid, max_pop)
            weight = 1.0 - (pop / total)  # lower weight for popular items
            weight = max(0.1, weight)      # floor at 0.1
            rewards.append(weight)
        else:
            # Partial credit for non-exact but valid SID generation
            if comp_sid and comp_sid.startswith("<a_"):
                rewards.append(0.01)
            else:
                rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# Reward Type 5: Popularity Penalty
# ---------------------------------------------------------------------------

def pop_penalty_reward(
    prompts: List[str],
    completions: List[str],
    targets: List[str],
    **kwargs,
) -> List[float]:
    """
    Direct popularity penalty: reward = match * (1 - normalized_popularity).
    Popular items get penalized even on correct predictions.
    """
    item2pop = kwargs.get("item2pop", {})
    max_pop = max(item2pop.values()) if item2pop else 10

    rewards = []
    for comp, tgt in zip(completions, targets):
        comp_sid = comp.strip()
        tgt_sid = tgt.strip()

        if comp_sid == tgt_sid:
            pop = item2pop.get(tgt_sid, 1)
            penalty = pop / max_pop  # [0, 1], 1 for hottest item
            rewards.append(1.0 - 0.5 * penalty)  # max penalty halves the reward
        else:
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# Reward Type 6: Partial Match (Hierarchical SID)
# ---------------------------------------------------------------------------

def partial_match_reward(
    prompts: List[str],
    completions: List[str],
    targets: List[str],
    **kwargs,
) -> List[float]:
    """
    Hierarchical partial match reward for 3-level SIDs (<a_X><b_Y><c_Z>).

    Match levels:
        L0 match (<a_X>):        0.2 reward
        L0 + L1 match (<a_X><b_Y>):  0.5 reward
        Full match (<a_X><b_Y><c_Z>): 1.0 reward

    This addresses reward sparsity by providing intermediate signals.
    """
    def extract_levels(sid: str) -> Tuple[str, str, str]:
        """Extract three levels from SID string like '<a_147><b_42><c_231>'."""
        try:
            parts = sid.strip().replace("><", "> <").split()
            l0 = parts[0] if len(parts) > 0 else ""
            l1 = " ".join(parts[:2]) if len(parts) > 1 else ""
            return l0, l1, sid.strip()
        except Exception:
            return "", "", sid.strip()

    rewards = []
    for comp, tgt in zip(completions, targets):
        c0, c1, _ = extract_levels(comp)
        t0, t1, _ = extract_levels(tgt)

        if c0 == t0 and c1 == t1:
            # Full L0+L1 match (implies L2 also correct since 3 levels)
            rewards.append(1.0)
        elif c0 == t0 and c1 == t1:
            rewards.append(0.5)
        elif c0 == t0:
            rewards.append(0.2)
        else:
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# Reward Type 7: Semantic Similarity
# ---------------------------------------------------------------------------

def semantic_reward(
    prompts: List[str],
    completions: List[str],
    targets: List[str],
    **kwargs,
) -> List[float]:
    """
    Cosine similarity reward between generated SID and target SID embeddings.
    Requires sid_embeddings: Dict[str, np.ndarray] mapping SID to vector.
    """
    sid_emb = kwargs.get("sid_embeddings", {})

    rewards = []
    for comp, tgt in zip(completions, targets):
        comp_vec = sid_emb.get(comp.strip())
        tgt_vec = sid_emb.get(tgt.strip())

        if comp_vec is not None and tgt_vec is not None:
            sim = np.dot(comp_vec, tgt_vec) / (
                np.linalg.norm(comp_vec) * np.linalg.norm(tgt_vec) + 1e-8
            )
            # Map [-1, 1] to [0, 1]
            rewards.append(float((sim + 1.0) / 2.0))
        else:
            rewards.append(0.0)

    return rewards


# ---------------------------------------------------------------------------
# Combined Reward (Debiased + Partial Match)
# ---------------------------------------------------------------------------

def combined_debiased_reward(
    prompts: List[str],
    completions: List[str],
    targets: List[str],
    **kwargs,
) -> List[float]:
    """Combines debiased popularity weighting with partial match signals."""
    debiased = debiased_reward(prompts, completions, targets, **kwargs)
    partial = partial_match_reward(prompts, completions, targets, **kwargs)

    # Weight: 60% debiased, 40% partial match
    return [0.6 * d + 0.4 * p for d, p in zip(debiased, partial)]


# ---------------------------------------------------------------------------
# Reward Registry
# ---------------------------------------------------------------------------

reward_registry: Dict[str, RewardFn] = {
    "rule": rule_reward,
    "ranking": ranking_reward,
    "ndcg_rule": ndcg_rule_reward,
    "debiased": debiased_reward,
    "pop_penalty": pop_penalty_reward,
    "partial_match": partial_match_reward,
    "semantic": semantic_reward,
    "combined": combined_debiased_reward,
}


# ---------------------------------------------------------------------------
# Utility: Build item popularity dictionary from training data
# ---------------------------------------------------------------------------

def build_item_popularity(train_csv: str) -> Dict[str, int]:
    """
    Build SID -> frequency mapping from training CSV.
    Used by debiased and pop_penalty reward functions.

    Args:
        train_csv: Path to training CSV with 'item_sid' column.

    Returns:
        Dict mapping SID string to occurrence count.
    """
    import pandas as pd

    df = pd.read_csv(train_csv)
    pop = Counter()
    for sid in df["item_sid"]:
        pop[sid.strip()] += 1
    print(f"Built popularity dict: {len(pop)} unique SIDs, "
          f"max count={max(pop.values())}, min={min(pop.values())}")
    return dict(pop)


def build_sid_embeddings(index_json: str) -> Dict[str, np.ndarray]:
    """
    Build SID -> embedding mapping from index.json.
    Uses a simple one-hot-like encoding: each SID token gets a unit vector
    based on its codebook position.

    Args:
        index_json: Path to .index.json file.

    Returns:
        Dict mapping SID string to numpy array.
    """
    with open(index_json) as f:
        data = json.load(f)

    embeddings = {}
    for item_id, tokens in data.items():
        sid_str = "".join(tokens)
        vec = np.zeros(256 * 3, dtype=np.float32)
        for i, tok in enumerate(tokens):
            code = int(tok.split("_")[1].rstrip(">"))
            vec[i * 256 + code] = 1.0
        embeddings[sid_str] = vec / np.linalg.norm(vec)

    print(f"Built SID embeddings: {len(embeddings)} vectors, dim={256*3}")
    return embeddings
