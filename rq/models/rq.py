import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,
                 # === EMA + Codebook Reset 参数 (透传到每个 VQ 层) ===
                 ema_decay=0.99,
                 dead_threshold=2.0,
                 ema_warmup_steps=100):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.ema_decay = ema_decay
        self.dead_threshold = dead_threshold
        self.ema_warmup_steps = ema_warmup_steps

        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters,
                                                        # 透传 EMA 参数
                                                        ema_decay=self.ema_decay,
                                                        dead_threshold=self.dead_threshold,
                                                        ema_warmup_steps=self.ema_warmup_steps)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def get_codebook_usage(self):
        """
        Aggregate codebook usage statistics across all VQ layers.

        Returns:
            dict mapping layer index (e.g. 'layer_0') to per-layer usage dict,
            plus a 'summary' key with averaged metrics.
        """
        all_usage = {}
        perplexities = []
        dead_counts = []
        dead_ratios = []

        for i, quantizer in enumerate(self.vq_layers):
            usage = quantizer.get_codebook_usage()
            key = f'layer_{i}'
            all_usage[key] = usage
            perplexities.append(usage['perplexity'])
            dead_counts.append(usage['dead_count'])
            dead_ratios.append(usage['dead_ratio'])

        all_usage['summary'] = {
            'perplexity_mean': sum(perplexities) / len(perplexities) if perplexities else 0.0,
            'dead_count_total': sum(dead_counts),
            'dead_ratio_mean': sum(dead_ratios) / len(dead_ratios) if dead_ratios else 0.0,
            'perplexities': perplexities,
            'dead_counts': dead_counts,
        }

        return all_usage

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices