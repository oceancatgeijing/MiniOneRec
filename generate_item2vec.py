#!/usr/bin/env python3
"""
generate_item2vec.py — Train Item2Vec Collaborative Embeddings from User Interaction Data
==========================================================================================

This script reads the MiniOneRec training CSV (user interaction sequences),
trains a Word2Vec model treating each user's item history as a "sentence" and
each Item ID as a "word", and outputs per-item collaborative embeddings.

Usage:
    python generate_item2vec.py \
        --train_csv data/Amazon/train/Office_Products_5_2016-10-2018-11.csv \
        --output data/Amazon/index/item_collaborative_emb.pt \
        --vector_size 64 --window 5 --negative 5 --epochs 10 --seed 42

Output:
    A PyTorch file containing:
      - embeddings: Dict[int, torch.Tensor]  (item_id -> 64-dim vector)
      - item_count: int
      - vector_size: int
      - cold_start_vector: torch.Tensor  (zero vector for unseen items)

Dependencies:
    pip install gensim torch pandas
"""

import argparse
import collections
import os
import sys
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_user_sequences(csv_path: str) -> List[List[int]]:
    """
    Parse training CSV and extract per-user item interaction sequences.

    Each row in the CSV represents one interaction: a user, their history,
    and the target item they interacted with next.

    We construct each user's full sequence as:
        history_item_ids + [target_item_id]

    For users with multiple rows, we concatenate all their interactions
    in chronological order (the CSV is ordered by time).

    Args:
        csv_path: Path to training CSV with columns:
                  user_id, history_item_title, item_title,
                  history_item_id, item_id, history_item_sid, item_sid

    Returns:
        List of sequences, where each sequence is List[int] of item IDs.
    """
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}, Users: {df['user_id'].nunique()}")

    # Group by user_id and aggregate all history + target items in order
    user_sequences = collections.defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing sequences"):
        user_id = row['user_id']

        # Parse history_item_id from Python list string, e.g. "[283]" -> [283]
        try:
            hist_ids = eval(str(row['history_item_id']))
        except Exception:
            hist_ids = []

        target_id = int(row['item_id'])

        # Extend the user's sequence: history + target
        user_sequences[user_id].extend(hist_ids)
        user_sequences[user_id].append(target_id)

    # Convert to list of sequences
    sequences = [list(seq) for seq in user_sequences.values()]

    # Report statistics
    all_items: Set[int] = set()
    total_interactions = 0
    for seq in sequences:
        all_items.update(seq)
        total_interactions += len(seq)

    print(f"  Users (sequences): {len(sequences)}")
    print(f"  Total interactions: {total_interactions}")
    print(f"  Unique items: {len(all_items)}")
    print(f"  Avg sequence length: {total_interactions / len(sequences):.1f}")

    return sequences


def load_user_sequences_simple(csv_path: str) -> List[List[str]]:
    """
    Parse training CSV (string-token version).

    Same as load_user_sequences but returns string tokens (e.g. "item_284")
    instead of integers — useful when item IDs are sparse and you want
    Word2Vec to treat them as categorical tokens.

    Args:
        csv_path: Path to training CSV.

    Returns:
        List of sequences, where each sequence is List[str] of string item IDs.
    """
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}, Users: {df['user_id'].nunique()}")

    user_sequences = collections.defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing sequences"):
        user_id = row['user_id']

        try:
            hist_ids = eval(str(row['history_item_id']))
        except Exception:
            hist_ids = []

        target_id = int(row['item_id'])

        # Store as string tokens for Word2Vec
        for hid in hist_ids:
            user_sequences[user_id].append(f"item_{hid}")
        user_sequences[user_id].append(f"item_{target_id}")

    sequences = [list(seq) for seq in user_sequences.values()]
    print(f"  Users: {len(sequences)}, Total interactions: {sum(len(s) for s in sequences)}")
    return sequences


# ---------------------------------------------------------------------------
# Item2Vec Training
# ---------------------------------------------------------------------------

class EpochLogger(CallbackAny2Vec):
    """Callback to log training progress each epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        loss = model.get_latest_training_loss()
        print(f"  Epoch {self.epoch}: loss = {loss:.2f}" if loss > 0
              else f"  Epoch {self.epoch} completed")


def train_item2vec(
    sequences: List[List[str]],
    vector_size: int = 64,
    window: int = 5,
    negative: int = 5,
    sg: int = 1,
    epochs: int = 10,
    min_count: int = 1,
    workers: int = 4,
    seed: int = 42,
) -> Word2Vec:
    """
    Train Item2Vec model using gensim Word2Vec (skip-gram with negative sampling).

    Treats each user's interaction sequence as a "sentence" and each item
    token as a "word". The learned embeddings capture collaborative patterns:
    items that frequently appear together in user histories get similar vectors.

    Args:
        sequences: List of item token sequences, e.g. [['item_283', 'item_284'], ...]
        vector_size: Dimension of the learned item embeddings.
        window: Maximum distance between current and predicted item within a sequence.
        negative: Number of negative samples for negative sampling (5-20 recommended).
        sg: Training algorithm: 1 = skip-gram, 0 = CBOW.
        epochs: Number of training epochs.
        min_count: Minimum frequency for an item to be included in the vocabulary.
        workers: Number of parallel workers.
        seed: Random seed for reproducibility.

    Returns:
        Trained Word2Vec model.
    """
    print(f"\nTraining Item2Vec model:")
    print(f"  vector_size={vector_size}, window={window}, negative={negative}")
    print(f"  sg={sg} (skip-gram), epochs={epochs}, workers={workers}")

    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        workers=workers,
        seed=seed,
        epochs=epochs,
        callbacks=[EpochLogger()],
    )

    vocab_size = len(model.wv)
    print(f"\nTraining complete. Vocabulary size: {vocab_size}")
    return model


# ---------------------------------------------------------------------------
# Embedding Extraction & Export
# ---------------------------------------------------------------------------

def extract_and_save_embeddings(
    model: Word2Vec,
    output_path: str,
    cold_start_default: str = "zero",
) -> Dict[int, torch.Tensor]:
    """
    Extract item embeddings from trained Word2Vec model and save as PyTorch file.

    Handles cold-start items (items not in training vocabulary) by assigning
    a zero vector.

    Args:
        model: Trained Word2Vec model.
        output_path: Path to save the .pt file.
        cold_start_default: Strategy for unseen items ('zero' = all-zero vector).

    Returns:
        Dict mapping item_id (int) to embedding tensor.
    """
    print(f"\nExtracting embeddings...")

    embeddings: Dict[int, torch.Tensor] = {}
    vocab = set(model.wv.index_to_key)

    for token in vocab:
        # Parse "item_284" -> 284
        item_id = int(token.replace("item_", ""))
        vec = torch.from_numpy(model.wv[token].copy()).float()
        embeddings[item_id] = vec

    # Determine cold-start vector
    vector_size = model.wv.vector_size
    if cold_start_default == "zero":
        zero_vec = torch.zeros(vector_size)
    else:
        raise ValueError(f"Unknown cold_start_default: {cold_start_default}")

    # Save
    save_dict = {
        "embeddings": embeddings,
        "item_count": len(embeddings),
        "vector_size": vector_size,
        "cold_start_vector": zero_vec,
        "vocab_size": len(vocab),
        "train_loss": model.get_latest_training_loss(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(save_dict, output_path)

    print(f"  Saved {len(embeddings)} item embeddings to: {output_path}")
    print(f"  Vector dimension: {vector_size}")
    print(f"  Cold-start strategy: {cold_start_default} (zero vector)")

    return embeddings


def load_embeddings(filepath: str) -> Dict[int, torch.Tensor]:
    """
    Load saved item collaborative embeddings.

    Args:
        filepath: Path to saved .pt file.

    Returns:
        Dict mapping item_id to embedding tensor.
    """
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    return data["embeddings"]


def get_embedding(
    item_id: int,
    embeddings: Dict[int, torch.Tensor],
    cold_start_vector: torch.Tensor = None,
) -> torch.Tensor:
    """
    Get embedding for a given item_id. Returns zero vector for cold-start items.

    Args:
        item_id: Item ID to look up.
        embeddings: Dict from item_id to embedding tensor.
        cold_start_vector: Vector for unseen items (defaults to zero).

    Returns:
        Embedding tensor of shape (vector_size,).
    """
    if item_id in embeddings:
        return embeddings[item_id]
    if cold_start_vector is not None:
        return cold_start_vector
    # Infer vector_size from first embedding
    for v in embeddings.values():
        return torch.zeros_like(v)
    return torch.zeros(64)  # fallback


# ---------------------------------------------------------------------------
# Statistics & Diagnostics
# ---------------------------------------------------------------------------

def print_statistics(
    model: Word2Vec,
    embeddings: Dict[int, torch.Tensor],
    topn: int = 10,
):
    """Print diagnostic statistics about the trained embeddings."""
    print(f"\n{'='*55}")
    print(f"  Item2Vec Training Statistics")
    print(f"{'='*55}")

    # Vocabulary
    print(f"  Vocabulary:          {len(model.wv):>8,}")
    print(f"  Embedding dim:       {model.wv.vector_size:>8}")

    # Embedding stats
    all_vecs = torch.stack(list(embeddings.values()))
    print(f"  Embedding norm mean: {all_vecs.norm(dim=1).mean():>8.4f}")
    print(f"  Embedding norm std:  {all_vecs.norm(dim=1).std():>8.4f}")

    # Similarity examples (random items)
    print(f"\n  Top-{topn} similar items (examples):")
    sample_items = list(embeddings.keys())[:5]
    for item_id in sample_items:
        token = f"item_{item_id}"
        if token not in model.wv:
            continue
        similar = model.wv.most_similar(token, topn=topn)
        sim_items = [f"item_{t.split('_')[1]}" for t, _ in similar]
        sim_scores = [f"{s:.3f}" for _, s in similar]
        print(f"    item_{item_id}: {', '.join(sim_items)}")
        print(f"      scores: {', '.join(sim_scores)}")

    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Item2Vec collaborative embeddings from user interaction data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training on Office_Products
  python generate_item2vec.py \\
      --train_csv data/Amazon/train/Office_Products_5_2016-10-2018-11.csv \\
      --output data/Amazon/index/item_collaborative_emb.pt

  # Custom hyperparameters
  python generate_item2vec.py \\
      --train_csv data/Amazon/train/Office_Products_5_2016-10-2018-11.csv \\
      --output data/Amazon/index/item_collaborative_emb.pt \\
      --vector_size 64 --window 10 --negative 10 --epochs 20
        """,
    )
    parser.add_argument(
        "--train_csv", required=True,
        help="Path to training CSV (MiniOneRec format with user_id, history_item_id, item_id)"
    )
    parser.add_argument(
        "--output", default="data/Amazon/index/item_collaborative_emb.pt",
        help="Output path for PyTorch embedding file (.pt)"
    )
    parser.add_argument(
        "--vector_size", type=int, default=64,
        help="Embedding dimension (default: 64)"
    )
    parser.add_argument(
        "--window", type=int, default=5,
        help="Context window size (default: 5)"
    )
    parser.add_argument(
        "--negative", type=int, default=5,
        help="Negative samples (default: 5)"
    )
    parser.add_argument(
        "--sg", type=int, default=1, choices=[0, 1],
        help="Training algorithm: 1=Skip-gram, 0=CBOW (default: 1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs (default: 10)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed similarity examples"
    )

    args = parser.parse_args()

    # ---- Load data ----
    sequences = load_user_sequences_simple(args.train_csv)

    # ---- Train Item2Vec ----
    model = train_item2vec(
        sequences=sequences,
        vector_size=args.vector_size,
        window=args.window,
        negative=args.negative,
        sg=args.sg,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )

    # ---- Extract & Save ----
    embeddings = extract_and_save_embeddings(
        model=model,
        output_path=args.output,
    )

    # ---- Statistics ----
    if args.verbose:
        print_statistics(model, embeddings)

    print("Done.")


if __name__ == "__main__":
    main()
