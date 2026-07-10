#!/usr/bin/env python3
"""
Generate Semantic IDs (SIDs) using an EMA-optimized RQ-VAE checkpoint.

Usage:
    python rq/generate_indices_ema.py \
        --ckpt_path output_dir/Jul-09-2026_16-48-32/epoch_199_collision_0.1911_model.pth \
        --data_path data/Amazon/index/Office_Products.emb-qwen-td.npy \
        --output data/Amazon/index/Office_Products_ema.index.json \
        --dataset Office_Products \
        --device cuda:0
"""

import collections
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import sys

# Add rq/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from datasets import EmbDataset, EmbDatasetWithCollab
from models.rqvae import RQVAE


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])
    return collision_item_groups


def main():
    parser = argparse.ArgumentParser(
        description="Generate SIDs from EMA-optimized RQ-VAE checkpoint")
    parser.add_argument("--ckpt_path", required=True,
                        help="Path to RQ-VAE checkpoint .pth file")
    parser.add_argument("--data_path", required=True,
                        help="Path to embedding .npy file")
    parser.add_argument("--output", required=True,
                        help="Output path for .index.json")
    parser.add_argument("--dataset", default="Office_Products",
                        help="Dataset name (for logging)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_collision_iters", type=int, default=20,
                        help="Max Sinkhorn collision-resolution iterations")
    parser.add_argument("--collab_emb_path", type=str, default=None,
                        help="Path to collaborative embedding .pt file. "
                             "If provided, text+collab embeddings are concatenated.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output: {args.output}")

    # ---- Load checkpoint ----
    ckpt = torch.load(args.ckpt_path, map_location=torch.device('cpu'),
                      weights_only=False)
    train_args = ckpt["args"]
    state_dict = ckpt["state_dict"]
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}, "
          f"collision_rate: {ckpt.get('best_collision_rate', 'N/A')}")

    # ---- Load data ----
    if args.collab_emb_path:
        print(f"Loading text + collaborative embeddings...")
        data = EmbDatasetWithCollab(args.data_path, args.collab_emb_path)
        collab_dim = data.collab_dim
    else:
        print(f"Loading text-only embeddings...")
        data = EmbDataset(args.data_path)
        collab_dim = 0
    print(f"Loaded {len(data)} embeddings, dim={data.dim}")

    # ---- Build model ----
    # Extract EMA params from checkpoint args (may not exist for older checkpoints)
    ema_decay = getattr(train_args, 'ema_decay', 0.99)
    dead_threshold = getattr(train_args, 'dead_threshold', 2.0)
    ema_warmup_steps = getattr(train_args, 'ema_warmup_steps', 100)
    # Preserve collab_dim if loading a collab-enhanced checkpoint
    saved_collab_dim = getattr(train_args, 'collab_dim', 0)
    if collab_dim > 0 and saved_collab_dim == 0:
        # We're adding collab to a checkpoint that was trained without it
        print(f"Note: Checkpoint was trained without collab embeddings. "
              f"collab_gate will be randomly initialized.")
        use_strict = False  # collab_gate won't exist in old checkpoint
    else:
        use_strict = True

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=train_args.num_emb_list,
        e_dim=train_args.e_dim,
        layers=train_args.layers,
        dropout_prob=train_args.dropout_prob,
        bn=train_args.bn,
        loss_type=train_args.loss_type,
        quant_loss_weight=train_args.quant_loss_weight,
        kmeans_init=train_args.kmeans_init,
        kmeans_iters=train_args.kmeans_iters,
        sk_epsilons=train_args.sk_epsilons,
        sk_iters=train_args.sk_iters,
        ema_decay=ema_decay,
        dead_threshold=dead_threshold,
        ema_warmup_steps=ema_warmup_steps,
        collab_dim=max(collab_dim, saved_collab_dim),
    )

    model.load_state_dict(state_dict, strict=use_strict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded. EMA decay={ema_decay}, "
          f"dead_threshold={dead_threshold}, warmup={ema_warmup_steps}, "
          f"collab_dim={model.collab_dim}")

    # ---- Encode all items ----
    data_loader = DataLoader(data, num_workers=getattr(train_args, 'num_workers', 4),
                             batch_size=64, shuffle=False, pin_memory=True)
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    all_indices = []
    all_indices_str = []

    for d in tqdm(data_loader, desc="Encoding"):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    print(f"Initial unique SIDs: {len(set(all_indices_str.tolist()))} / {len(all_indices_str)}")
    print(f"Initial collision rate: {1 - len(set(all_indices_str.tolist())) / len(all_indices_str):.4f}")

    # ---- Collision resolution with Sinkhorn ----
    # Enable Sinkhorn for last layer if not already
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    while True:
        if tt >= args.max_collision_iters or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(f"Iteration {tt+1}: {len(collision_item_groups)} collision groups")
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    # ---- Statistics ----
    tot_item = len(all_indices_str)
    unique_sids = len(set(all_indices_str.tolist()))
    collision_rate = (tot_item - unique_sids) / tot_item
    print(f"\nFinal Statistics:")
    print(f"  Total items:     {tot_item}")
    print(f"  Unique SIDs:     {unique_sids}")
    print(f"  Collision rate:  {collision_rate:.4f} ({collision_rate*100:.2f}%)")
    print(f"  Max conflicts:   {max(get_indices_count(all_indices_str).values())}")

    # ---- Save ----
    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[str(item)] = list(indices)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fp:
        json.dump(all_indices_dict, fp, indent=2)
    print(f"Saved: {args.output} ({len(all_indices_dict)} entries)")

    return collision_rate


if __name__ == "__main__":
    main()
