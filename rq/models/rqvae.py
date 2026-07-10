import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 # === EMA + Codebook Reset ===
                 ema_decay=0.99,
                 dead_threshold=2.0,
                 ema_warmup_steps=100,
                 # === Collaborative Embedding Fusion ===
                 collab_dim=0,
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.collab_dim = collab_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.ema_decay = ema_decay
        self.dead_threshold = dead_threshold
        self.ema_warmup_steps = ema_warmup_steps

        # Learnable per-dimension gate for collaborative signal fusion.
        # Gate is a vector of shape (collab_dim,); after sigmoid, each element
        # controls how much of the corresponding collab dimension is used.
        # Initialized to ~0.5 so training starts with balanced mix.
        if self.collab_dim > 0:
            self.collab_gate = nn.Parameter(torch.zeros(self.collab_dim))
            self.text_dim = self.in_dim - self.collab_dim
        else:
            self.collab_gate = None
            self.text_dim = self.in_dim

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          ema_decay=self.ema_decay,
                                          dead_threshold=self.dead_threshold,
                                          ema_warmup_steps=self.ema_warmup_steps,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        # ---- Collaborative gating (applied before encoder) ----
        if self.collab_gate is not None:
            text_part = x[:, :self.text_dim]              # (B, text_dim)
            collab_part = x[:, self.text_dim:]            # (B, collab_dim)
            gate_active = torch.sigmoid(self.collab_gate) # (collab_dim,)
            collab_gated = gate_active * collab_part      # per-dim gating
            x = torch.cat([text_part, collab_gated], dim=-1)

        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        # Apply gating in eval mode too (uses current learned gate values)
        if self.collab_gate is not None:
            text_part = xs[:, :self.text_dim]
            collab_part = xs[:, self.text_dim:]
            gate_active = torch.sigmoid(self.collab_gate)
            collab_gated = gate_active * collab_part
            xs = torch.cat([text_part, collab_gated], dim=-1)

        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def get_gate_stats(self):
        """Return the current gate activation values for monitoring."""
        if self.collab_gate is None:
            return None
        with torch.no_grad():
            gate_active = torch.sigmoid(self.collab_gate)
        return {
            'gate_min': gate_active.min().item(),
            'gate_mean': gate_active.mean().item(),
            'gate_max': gate_active.max().item(),
            'gate_vector': gate_active.cpu().numpy(),
        }

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon