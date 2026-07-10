import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA (Exponential Moving Average) codebook updates
    and dead code reset mechanism.

    References:
        VQ-VAE-2 (Razavi et al., 2019): https://arxiv.org/abs/1906.00446
        Improved VQGAN (Yu et al., 2021): https://arxiv.org/abs/2110.04627
    """

    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon=0.003, sk_iters=100,
                 # === EMA + Codebook Reset 参数 ===
                 ema_decay=0.99,
                 dead_threshold=2.0,
                 ema_warmup_steps=100,
                 reset_noise_scale=1e-4):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        # EMA 参数
        self.ema_decay = ema_decay
        self.dead_threshold = dead_threshold
        self.ema_warmup_steps = ema_warmup_steps
        self.reset_noise_scale = reset_noise_scale

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        # EMA tracking buffers: persist across training steps & checkpoint save/load
        # _ema_cluster_size: (n_e,)  — running count of assignments per code
        # _ema_embed_sum:    (n_e, e_dim) — running sum of assigned encoder outputs
        self.register_buffer('_ema_cluster_size', torch.zeros(self.n_e))
        self.register_buffer('_ema_embed_sum', torch.zeros(self.n_e, self.e_dim))
        self.register_buffer('_ema_initialized', torch.tensor(False))
        self.register_buffer('_train_step_count', torch.tensor(0, dtype=torch.long))

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    # =====================================================================
    # EMA Codebook Update + Dead Code Reset
    # =====================================================================

    @torch.no_grad()
    def _ema_update(self, x_flat, indices_onehot):
        """
        Update codebook using Exponential Moving Average (EMA).

        Standard VQ-VAE-2 formulation:
            N_i^(t) = γ · N_i^(t-1) + (1-γ) · n_i^(t)
            m_i^(t) = γ · m_i^(t-1) + (1-γ) · sum_i^(t)
            e_i^(t) = m_i^(t) / (N_i^(t) + ε)

        Args:
            x_flat:     (B, D) encoder outputs for current batch
            indices_onehot: (B, K) one-hot assignment matrix.
                          For hard assignment (argmin): one-hot rows
                          For soft assignment (sinkhorn): Q soft matrix
        """
        B, D = x_flat.shape
        K = self.n_e

        # --- 1. Compute current-batch statistics ---
        # n_i: per-code assignment count (soft or hard)
        # sum_i: per-code sum of assigned encoder outputs
        n_i = indices_onehot.sum(dim=0)                    # (K,)
        sum_i = indices_onehot.t().float() @ x_flat.float()  # (K, D)

        # --- 2. Initialize or update EMA buffers ---
        if not self._ema_initialized:
            # First call: initialize directly with batch statistics
            self._ema_cluster_size.copy_(n_i)
            self._ema_embed_sum.copy_(sum_i)
            self._ema_initialized.fill_(True)
        else:
            decay = self.ema_decay
            one_minus_decay = 1.0 - decay
            self._ema_cluster_size.mul_(decay).add_(n_i, alpha=one_minus_decay)
            self._ema_embed_sum.mul_(decay).add_(sum_i, alpha=one_minus_decay)

        # --- 3. Update codebook with Laplace-smoothed EMA ---
        eps = 1e-5
        # Avoid division by very small numbers
        laplace_smoothed = self._ema_cluster_size + eps
        updated_codebook = self._ema_embed_sum / laplace_smoothed.unsqueeze(-1)

        # Copy EMA result into embedding weights
        self.embedding.weight.data.copy_(updated_codebook.to(self.embedding.weight.dtype))

        # --- 4. Dead code detection & reset ---
        self._reset_dead_codes(x_flat)

        # --- 5. Increment step counter ---
        self._train_step_count.add_(1)

    @torch.no_grad()
    def _reset_dead_codes(self, x_flat):
        """
        Detect and reset dead codes — codebook entries whose EMA cluster size
        falls below the threshold.

        For each dead code:
        1. Sample a random encoder output from the current batch
        2. Reinitialize the embedding to that vector + small noise
        3. Reset EMA statistics for that entry

        Anti-jitter safeguards:
        - Skip during warmup (first `ema_warmup_steps` training steps)
        - Cap maximum reset count per step to 10% of codebook size
        """
        # Guard: skip during warmup
        if self._train_step_count < self.ema_warmup_steps:
            return

        # Guard: EMA must be initialized
        if not self._ema_initialized:
            return

        # Detect dead codes
        dead_mask = self._ema_cluster_size < self.dead_threshold  # (K,) bool
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]        # (num_dead,) int64
        num_dead = dead_indices.numel()

        if num_dead == 0:
            return

        # Cap: reset at most 10% of codebook per step
        max_reset = max(1, int(self.n_e * 0.10))
        if num_dead > max_reset:
            # Prioritize the most dead codes (smallest cluster size)
            dead_sizes = self._ema_cluster_size[dead_indices]
            _, sorted_idx = torch.sort(dead_sizes)
            dead_indices = dead_indices[sorted_idx[:max_reset]]
            num_dead = max_reset

        B = x_flat.shape[0]

        for idx in dead_indices:
            # Pick a random encoder output from current batch
            rand_idx = torch.randint(0, B, (1,), device=x_flat.device).item()
            new_emb = x_flat[rand_idx].clone()

            # Add small noise to prevent overfitting to one sample
            noise = torch.randn_like(new_emb) * self.reset_noise_scale
            new_emb = new_emb + noise

            # Copy into embedding
            self.embedding.weight.data[idx].copy_(
                new_emb.to(self.embedding.weight.dtype)
            )

            # Reset EMA statistics for this code
            self._ema_cluster_size[idx] = 1.0
            self._ema_embed_sum[idx].copy_(
                new_emb.to(self._ema_embed_sum.dtype)
            )

    def get_codebook_usage(self):
        """
        Return codebook usage statistics for monitoring.

        Returns:
            dict with keys:
                'perplexity':  exp(entropy) — higher is better, max = n_e
                'dead_count':  number of codes with EMA size < threshold
                'dead_ratio':  dead_count / n_e
                'usage_min':   min EMA cluster size
                'usage_mean':  mean EMA cluster size
                'usage_max':   max EMA cluster size
        """
        if not self._ema_initialized:
            return {
                'perplexity': 0.0,
                'dead_count': 0,
                'dead_ratio': 0.0,
                'codebook_size': self.n_e,
                'usage_min': 0.0,
                'usage_mean': 0.0,
                'usage_max': 0.0,
            }

        sizes = self._ema_cluster_size.float()
        total = sizes.sum()
        if total < 1e-8:
            probs = torch.ones_like(sizes) / self.n_e
        else:
            probs = sizes / total

        # Perplexity = exp(entropy)
        # entropy = - Σ p_i log(p_i), excluding zero-probability entries
        log_probs = torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs))
        entropy = -(probs * log_probs).sum()
        perplexity = torch.exp(entropy).item()

        dead_mask = sizes < self.dead_threshold
        dead_count = dead_mask.sum().item()

        return {
            'perplexity': perplexity,
            'dead_count': dead_count,
            'dead_ratio': dead_count / self.n_e,
            'codebook_size': self.n_e,
            'usage_min': sizes.min().item(),
            'usage_mean': sizes.mean().item(),
            'usage_max': sizes.max().item(),
        }

    # =====================================================================

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients (straight-through estimator)
        x_q = x + (x_q - x).detach()

        # ---- EMA codebook update (training only, no gradient) ----
        if self.training and self.ema_decay > 0.0:
            with torch.no_grad():
                # Build one-hot assignment matrix for EMA
                # For argmin (hard): one-hot encoding
                # For sinkhorn (soft): use Q directly
                if use_sk and self.sk_epsilon > 0:
                    # Q already exists from sinkhorn above, use as soft assignment
                    indices_onehot = Q.to(dtype=torch.float32)  # (B, K)
                else:
                    # Hard assignment: one-hot from argmin indices
                    indices_onehot = F.one_hot(
                        indices, num_classes=self.n_e
                    ).float()  # (B, K)

                self._ema_update(latent, indices_onehot)

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


