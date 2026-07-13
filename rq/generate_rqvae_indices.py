#!/usr/bin/env python3
"""Generate a semantic-ID index from a trained text-only RQ-VAE checkpoint."""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EmbDataset
from models.rqvae import RQVAE


def build_model(checkpoint_args, input_dim):
    return RQVAE(
        in_dim=input_dim,
        num_emb_list=checkpoint_args.num_emb_list,
        e_dim=checkpoint_args.e_dim,
        layers=checkpoint_args.layers,
        dropout_prob=checkpoint_args.dropout_prob,
        bn=checkpoint_args.bn,
        loss_type=checkpoint_args.loss_type,
        quant_loss_weight=checkpoint_args.quant_loss_weight,
        beta=getattr(checkpoint_args, "beta", 0.25),
        kmeans_init=checkpoint_args.kmeans_init,
        kmeans_iters=checkpoint_args.kmeans_iters,
        sk_epsilons=checkpoint_args.sk_epsilons,
        sk_iters=checkpoint_args.sk_iters,
        ema_decay=getattr(checkpoint_args, "ema_decay", 0.0),
        dead_threshold=getattr(checkpoint_args, "dead_threshold", 2.0),
        ema_warmup_steps=getattr(checkpoint_args, "ema_warmup_steps", 100),
        collab_dim=0,
    )


def code_statistics(codes, codebook_sizes):
    paths = [tuple(int(value) for value in row) for row in codes]
    metrics = {
        "item_count": len(paths),
        "unique_paths": len(set(paths)),
        "collision_rate": 1.0 - len(set(paths)) / max(len(paths), 1),
        "levels": [],
    }
    for level, size in enumerate(codebook_sizes):
        unique = len(set(int(row[level]) for row in codes))
        metrics["levels"].append({
            "level": level,
            "used_codes": unique,
            "codebook_size": int(size),
            "utilization": unique / max(int(size), 1),
        })
    return metrics


def collision_groups(codes):
    groups = {}
    for item_id, row in enumerate(codes):
        groups.setdefault(tuple(int(value) for value in row), []).append(item_id)
    return [item_ids for item_ids in groups.values() if len(item_ids) > 1]


def resolve_collisions(model, dataset, codes, device, max_rounds):
    """Reuse the original Sinkhorn collision-resolution path when needed."""
    if max_rounds <= 0:
        return codes, 0
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    rounds = 0
    for rounds in range(1, max_rounds + 1):
        groups = collision_groups(codes)
        if not groups:
            return codes, rounds - 1
        for item_ids in groups:
            batch = dataset[item_ids].to(device)
            indices = model.get_indices(batch, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item_id, row in zip(item_ids, indices):
                codes[item_id] = row
    return codes, rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metrics_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--collision_rounds", type=int, default=20)
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["args"]
    if getattr(checkpoint_args, "collab_emb_path", None):
        raise ValueError("This experiment requires a text-only RQ-VAE checkpoint")
    if float(getattr(checkpoint_args, "ema_decay", 0.0)) != 0.0:
        raise ValueError("This experiment requires EMA disabled (--ema_decay 0.0)")

    dataset = EmbDataset(args.data_path)
    model = build_model(checkpoint_args, dataset.dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    all_codes = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating RQ-VAE SIDs"):
            indices = model.get_indices(batch.to(args.device), use_sk=False)
            all_codes.append(indices.view(-1, indices.shape[-1]).cpu().numpy())
    codes = np.concatenate(all_codes, axis=0).astype(np.int64)
    raw_metrics = code_statistics(codes, checkpoint_args.num_emb_list)
    codes, resolution_rounds = resolve_collisions(
        model, dataset, codes, args.device, args.collision_rounds
    )

    prefixes = "abcde"
    index = {
        str(item_id): [f"<{prefixes[level]}_{int(code)}>" for level, code in enumerate(row)]
        for item_id, row in enumerate(codes)
    }
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)

    metrics = code_statistics(codes, checkpoint_args.num_emb_list)
    metrics["method"] = "rqvae"
    metrics["raw_collision_rate"] = raw_metrics["collision_rate"]
    metrics["collision_resolution_rounds"] = resolution_rounds
    metrics_path = args.metrics_path or os.path.splitext(args.output_path)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote index to {args.output_path}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
