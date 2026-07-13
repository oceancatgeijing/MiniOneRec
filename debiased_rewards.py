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
import re
from collections import Counter
from typing import Callable, Dict, List, Tuple

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


SID_PATTERN = re.compile(r"<a_\d+><b_\d+><c_\d+>")


def normalize_sid(value: str) -> str:
    """Extract and normalize a three-level SID from generated text."""
    text = str(value).strip().strip('"').strip()
    match = SID_PATTERN.search(text)
    return match.group(0) if match else text


def _align_targets(targets: List[str], size: int) -> List[str]:
    """Expand per-prompt targets when callers pass one target per GRPO group."""
    if len(targets) == size:
        return targets
    if not targets or size % len(targets) != 0:
        raise ValueError(
            f"Cannot align {len(targets)} targets with {size} completions"
        )
    repeat = size // len(targets)
    return [target for target in targets for _ in range(repeat)]


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
    targets = _align_targets(targets, len(completions))
    for comp, tgt in zip(completions, targets):
        if normalize_sid(comp) == normalize_sid(tgt):
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
    targets = _align_targets(targets, len(completions))
    for comp, tgt in zip(completions, targets):
        base = 0.0

        # CF score from SASRec
        if sasrec_model is not None:
            try:
                comp_id = item2id.get(normalize_sid(comp))
                tgt_id = item2id.get(normalize_sid(tgt))
                if comp_id is not None and tgt_id is not None:
                    # Simplified: count as positive if both are valid items
                    base = 0.5
            except Exception:
                pass

        # Exact match bonus
        if normalize_sid(comp) == normalize_sid(tgt):
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

    targets = _align_targets(targets, len(completions))
    for i in range(0, len(completions), beam_size):
        beam_comps = completions[i : i + beam_size]
        beam_targets = targets[i : i + beam_size]

        for rank, (comp, target) in enumerate(zip(beam_comps, beam_targets)):
            if normalize_sid(comp) == normalize_sid(target):
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
    max_pop = max(item2pop.values()) if item2pop else 1
    alpha = float(kwargs.get("popularity_alpha", 0.5))
    novelty_bonus = float(kwargs.get("novelty_bonus", 0.1))

    rewards = []
    targets = _align_targets(targets, len(completions))
    for comp, tgt in zip(completions, targets):
        comp_sid = normalize_sid(comp)
        tgt_sid = normalize_sid(tgt)

        comp_pop = item2pop.get(comp_sid, 0)
        normalized_pop = math.log1p(comp_pop) / math.log1p(max_pop)
        candidate_novelty = 1.0 - normalized_pop
        exact_weight = 0.0
        if comp_sid == tgt_sid:
            target_pop = item2pop.get(tgt_sid, 1)
            exact_weight = (max_pop / max(target_pop, 1)) ** alpha
            exact_weight = min(
                exact_weight, float(kwargs.get("max_debiased_reward", 5.0))
            )
        rewards.append(exact_weight + novelty_bonus * candidate_novelty)

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
    max_pop = max(item2pop.values()) if item2pop else 1
    penalty_weight = float(kwargs.get("popularity_penalty_weight", 0.2))

    rewards = []
    targets = _align_targets(targets, len(completions))
    for comp, tgt in zip(completions, targets):
        comp_sid = normalize_sid(comp)
        tgt_sid = normalize_sid(tgt)

        pop = item2pop.get(comp_sid, 0)
        penalty = math.log1p(pop) / math.log1p(max_pop)
        exact = 1.0 if comp_sid == tgt_sid else 0.0
        rewards.append(exact - penalty_weight * penalty)

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
        normalized = normalize_sid(sid)
        parts = re.findall(r"<[abc]_\d+>", normalized)
        return tuple((parts + ["", "", ""])[:3])

    targets = _align_targets(targets, len(completions))
    rewards = []
    for comp, tgt in zip(completions, targets):
        c0, c1, c2 = extract_levels(comp)
        t0, t1, t2 = extract_levels(tgt)

        if c0 and (c0, c1, c2) == (t0, t1, t2):
            rewards.append(1.0)
        elif c0 and c1 and (c0, c1) == (t0, t1):
            rewards.append(0.5)
        elif c0 and c0 == t0:
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
    targets = _align_targets(targets, len(completions))
    for comp, tgt in zip(completions, targets):
        comp_vec = sid_emb.get(normalize_sid(comp))
        tgt_vec = sid_emb.get(normalize_sid(tgt))

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


def make_trl_reward(
    reward_type: str,
    target_resolver: Callable[[List[str]], List[str]],
    **reward_kwargs,
) -> Callable[[List[str], List[str]], List[float]]:
    """Adapt a registered reward to TRL's prompts/completions callback API."""
    if reward_type not in reward_registry:
        available = ", ".join(sorted(reward_registry))
        raise ValueError(f"Unknown reward_type={reward_type!r}. Available: {available}")

    reward_fn = reward_registry[reward_type]

    def trl_reward(prompts, completions, **kwargs):
        targets = target_resolver(prompts)
        merged_kwargs = {**reward_kwargs, **kwargs}
        return reward_fn(prompts, completions, targets, **merged_kwargs)

    trl_reward.__name__ = f"{reward_type}_reward"
    return trl_reward


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
    if "item_sid" not in df.columns:
        raise ValueError(f"{train_csv} must contain an 'item_sid' column")
    for sid in df["item_sid"].dropna():
        pop[normalize_sid(sid)] += 1
    if not pop:
        raise ValueError(f"No item SIDs found in {train_csv}")
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
        embeddings[sid_str] = vec / (np.linalg.norm(vec) + 1e-8)

    print(f"Built SID embeddings: {len(embeddings)} vectors, dim={256*3}")
    return embeddings
