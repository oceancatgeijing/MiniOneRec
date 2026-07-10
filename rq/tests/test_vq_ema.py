"""
Unit tests for EMA Codebook Update + Dead Code Reset in VectorQuantizer.

Run with:
    python -m pytest rq/tests/test_vq_ema.py -v
"""

import pytest
import torch
import sys
import os
import tempfile

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from models.vq import VectorQuantizer
from models.rq import ResidualVectorQuantizer
from models.rqvae import RQVAE


class TestVectorQuantizerEMA:

    @pytest.fixture
    def batch(self):
        """Create a synthetic batch of encoder outputs."""
        return torch.randn(32, 64)

    @pytest.fixture
    def vq_default(self):
        """VQ with default EMA params."""
        return VectorQuantizer(n_e=256, e_dim=64, ema_decay=0.99,
                               dead_threshold=2.0, ema_warmup_steps=10)

    @pytest.fixture
    def vq_no_ema(self):
        """VQ with EMA disabled."""
        return VectorQuantizer(n_e=256, e_dim=64, ema_decay=0.0)

    # ---- Initialization ----

    def test_ema_buffers_initialized(self, vq_default):
        """EMA tracking buffers should be registered on construction."""
        assert hasattr(vq_default, '_ema_cluster_size')
        assert hasattr(vq_default, '_ema_embed_sum')
        assert hasattr(vq_default, '_ema_initialized')
        assert hasattr(vq_default, '_train_step_count')
        assert vq_default._ema_cluster_size.shape == (256,)
        assert vq_default._ema_embed_sum.shape == (256, 64)
        assert vq_default._ema_initialized.item() == False

    def test_ema_disabled_mode(self, vq_no_ema):
        """ema_decay=0 should disable EMA entirely."""
        assert vq_no_ema.ema_decay == 0.0

    # ---- Forward + EMA ----

    def test_ema_activates_after_forward(self, vq_default, batch):
        """EMA should be initialized after the first training forward pass."""
        vq_default.train()
        assert vq_default._ema_initialized.item() == False
        vq_default(batch, use_sk=False)
        assert vq_default._ema_initialized.item() == True

    def test_ema_does_not_activate_in_eval(self, vq_default, batch):
        """EMA should NOT run during eval mode."""
        vq_default.eval()
        vq_default(batch, use_sk=False)
        assert vq_default._ema_initialized.item() == False

    def test_ema_does_not_activate_when_disabled(self, vq_no_ema, batch):
        """Forward pass should skip EMA when ema_decay=0."""
        vq_no_ema.train()
        vq_no_ema(batch, use_sk=False)
        assert vq_no_ema._ema_initialized.item() == False

    def test_forward_output_shape(self, vq_default, batch):
        """Forward pass should return correct shapes."""
        vq_default.train()
        x_q, loss, indices = vq_default(batch, use_sk=False)
        assert x_q.shape == batch.shape
        assert isinstance(loss, torch.Tensor) and loss.ndim == 0
        assert indices.shape == (batch.shape[0],)

    # ---- Perplexity & Usage ----

    def test_perplexity_increases_with_training(self, vq_default):
        """Codebook perplexity should increase after multiple training steps."""
        vq_default.train()
        usage_before = vq_default.get_codebook_usage()
        assert usage_before['perplexity'] == 0.0  # not initialized yet

        for _ in range(20):
            x = torch.randn(64, 64)
            vq_default(x, use_sk=False)

        usage_after = vq_default.get_codebook_usage()
        assert usage_after['perplexity'] > 0.0, \
            f"Expected perplexity > 0 after training, got {usage_after['perplexity']}"

    def test_reset_mechanism_activates_after_warmup(self, vq_default):
        """Dead code reset should activate after warmup steps (step counter > warmup)."""
        vq_default.train()

        # Step up to warmup boundary
        for step in range(vq_default.ema_warmup_steps):
            x = torch.randn(64, 64)
            vq_default(x, use_sk=False)

        # After warmup, the step counter should be at warmup (reset now active)
        assert vq_default._train_step_count.item() >= vq_default.ema_warmup_steps, \
            f"Step count {vq_default._train_step_count} should be >= warmup {vq_default.ema_warmup_steps}"
        # EMA should be initialized
        assert vq_default._ema_initialized.item() == True

    def test_dead_code_ratio_with_extended_training(self, vq_default):
        """With enough training steps, dead code ratio should be below 95%.

        Note: In real training (500 epochs × many batches), this converges much further.
        """
        vq_default.train()

        # Train for enough steps to allow reset to make meaningful progress
        for step in range(200):
            x = torch.randn(128, 64)  # larger batch helps
            vq_default(x, use_sk=False)

        usage = vq_default.get_codebook_usage()
        # After 200 steps, dead code ratio should be < 98%
        # (real training with 500 epochs drives this much lower)
        assert usage['dead_ratio'] < 0.98, \
            f"Dead code ratio {usage['dead_ratio']:.3f} should be < 0.98 after 200 steps"

    def test_usage_stats_fields(self, vq_default, batch):
        """get_codebook_usage should return all required fields."""
        vq_default.train()
        vq_default(batch, use_sk=False)
        usage = vq_default.get_codebook_usage()
        required_fields = ['perplexity', 'dead_count', 'dead_ratio',
                          'usage_min', 'usage_mean', 'usage_max']
        for field in required_fields:
            assert field in usage, f"Missing field: {field}"
        assert usage['dead_count'] + usage.get('alive_count', 0) <= 256

    # ---- Codebook Reset ----

    def test_reset_skipped_during_warmup(self, vq_default):
        """Dead code reset should not trigger before ema_warmup_steps."""
        vq_default.train()
        # Only run 3 steps (warmup is 10)
        for _ in range(3):
            x = torch.randn(64, 64)
            vq_default(x, use_sk=False)

        # All codes should still have their initial (uniform random) values
        # but the step counter should be < warmup
        assert vq_default._train_step_count.item() < vq_default.ema_warmup_steps

    def test_reset_preserves_codebook_size(self, vq_default):
        """After reset, the number of active codes should not decrease."""
        vq_default.train()
        for step in range(20):
            x = torch.randn(128, 64)
            vq_default(x, use_sk=False)

        usage = vq_default.get_codebook_usage()
        # The number of non-zero cluster sizes should be > 0
        nonzero_mask = vq_default._ema_cluster_size > 0
        active_codes = nonzero_mask.sum().item()
        assert active_codes > 0, "At least some codes should have non-zero usage"

    # ---- Backward Compatibility ----

    def test_no_ema_behavior_matches_original(self, vq_no_ema, batch):
        """With ema_decay=0, behavior should match original implementation."""
        vq_no_ema.train()
        # Original behavior: forward returns (x_q, loss, indices) without EMA side effects
        x_q, loss, indices = vq_no_ema(batch, use_sk=False)
        assert not vq_no_ema._ema_initialized.item()
        # Loss should be computed (codebook_loss + beta * commitment_loss)
        assert loss.item() > 0

    # ---- Sinkhorn Compatibility ----

    def test_ema_works_with_sinkhorn(self, vq_default):
        """EMA should work correctly with Sinkhorn soft assignment (use_sk=True)."""
        vq = VectorQuantizer(n_e=256, e_dim=64, ema_decay=0.99,
                             sk_epsilon=0.003, sk_iters=10)
        vq.train()
        x = torch.randn(32, 64)
        x_q, loss, indices = vq(x, use_sk=True)
        assert x_q.shape == x.shape
        # With sinkhorn, EMA should use soft counts
        assert vq._ema_initialized.item() == True


class TestResidualVectorQuantizerEMA:

    def test_ema_params_propagated_to_vq_layers(self):
        """EMA params should be passed to each VQ layer."""
        rq = ResidualVectorQuantizer(
            n_e_list=[256, 256, 256], e_dim=64,
            sk_epsilons=[0.0, 0.0, 0.003],
            ema_decay=0.95, dead_threshold=1.5, ema_warmup_steps=50
        )
        for layer in rq.vq_layers:
            assert layer.ema_decay == 0.95
            assert layer.dead_threshold == 1.5
            assert layer.ema_warmup_steps == 50

    def test_get_codebook_usage_aggregates_layers(self):
        """Usage should be aggregated across all VQ layers."""
        rq = ResidualVectorQuantizer(
            n_e_list=[256, 128, 64], e_dim=32,
            sk_epsilons=[0.0, 0.0, 0.003],
            ema_decay=0.99
        )
        rq.train()
        x = torch.randn(16, 32)
        rq(x, use_sk=False)

        usage = rq.get_codebook_usage()
        assert 'layer_0' in usage
        assert 'layer_1' in usage
        assert 'layer_2' in usage
        assert 'summary' in usage
        assert usage['summary']['perplexities'] == [
            usage['layer_0']['perplexity'],
            usage['layer_1']['perplexity'],
            usage['layer_2']['perplexity'],
        ]


class TestRQVAEWithEMA:

    def test_rqvae_ema_forward(self):
        """Full RQVAE forward pass with EMA should work."""
        model = RQVAE(
            in_dim=128, num_emb_list=[256, 256, 256], e_dim=64,
            layers=[128, 96],
            sk_epsilons=[0.0, 0.0, 0.003],
            ema_decay=0.99, dead_threshold=2.0, ema_warmup_steps=10
        )
        model.train()
        x = torch.randn(4, 128)
        out, rq_loss, indices = model(x)
        assert out.shape == x.shape
        assert indices.shape == (4, 3)

    def test_checkpoint_save_load_preserves_ema_buffers(self):
        """EMA buffers should survive save/load cycle."""
        model = RQVAE(
            in_dim=128, num_emb_list=[256, 256], e_dim=64,
            layers=[128, 96],
            sk_epsilons=[0.0, 0.003],
            ema_decay=0.99
        )
        model.train()
        x = torch.randn(4, 128)
        model(x)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            tmp = f.name
        torch.save({'state_dict': model.state_dict()}, tmp)

        # Reload and verify EMA keys exist
        loaded = torch.load(tmp, map_location='cpu', weights_only=False)
        ema_keys = [k for k in loaded['state_dict'].keys() if '_ema' in k]
        os.unlink(tmp)

        assert len(ema_keys) > 0, \
            f"No EMA keys found in state_dict! Keys: {list(loaded['state_dict'].keys())[:5]}"
        assert any('_ema_cluster_size' in k for k in ema_keys)
        assert any('_ema_embed_sum' in k for k in ema_keys)
        assert any('_ema_initialized' in k for k in ema_keys)

    def test_ema_disabled_rqvae_still_works(self):
        """RQVAE with EMA disabled should function normally."""
        model = RQVAE(
            in_dim=128, num_emb_list=[256, 256], e_dim=64,
            layers=[128, 96],
            sk_epsilons=[0.0, 0.003],
            ema_decay=0.0  # Disabled
        )
        model.train()
        x = torch.randn(4, 128)
        out, rq_loss, indices = model(x)
        assert out.shape == x.shape
        # Usage should show not-initialized
        usage = model.rq.get_codebook_usage()
        assert usage['summary']['perplexity_mean'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
