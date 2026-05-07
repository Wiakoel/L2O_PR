"""
network/Spectral_Aggregation.py — Fingerprint-Aware Feature Aggregation (FAFA, §4.2)

Paper reference: Structural Fingerprint-Aware Attention (SFAA) — aggregation stage.

Components:
  - SpectralRingDecomposition: Prior-guided dynamic spectral ring decomposition
    with learnable DC floor and frequency calibration (Eq. 5).
  - RingQueryAttentionBlock: Proposal query self-attention + ring cross-attention (Eq. 6).
  - SpectralRingQueryAggregation: Multi-layer FAFA head producing global descriptors
    and fingerprint-aware attention maps (Eqs. 5-7).
    Optional confidence-aware token modulation:
      joint confidence = visibility_conf × spectral_conf applied as
      residual scaling before ring query blocks.

RESEARCH CORE — do not modify forward math paths.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralRingDecomposition(nn.Module):
    """Prior-Guided Dynamic Spectral Ring Decomposition with residual gating.

    Pipeline (fixed):
      1) FFT -> magnitude (DC preserves sign, AC absolute)
      2) Per-frequency affine calibration (gamma, beta)
      3) Residual gating: logits = prior_log_weights + tanh(content_gate_mlp(content))
      4) softmax(logits) weighted sum across frequencies
      5) Optional mask for ring saliency

    Learnable DC energy floor via softplus parametrization.
    Gate MLP last layer init to 0 -> delta=0 at init (stability first).
    """

    def __init__(self, num_freqs: int = 8, eps: float = 1e-6, gate_hidden: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        self.eps = eps

        # Learnable DC floor: softplus(-13.8) ~ 1e-6 (original behavior)
        self.dc_floor_logit = nn.Parameter(torch.tensor(-13.8))

        # Prior (static weights)
        self.prior_log_weights = nn.Parameter(self._get_init_weights())

        # Per-frequency affine calibration
        self.freq_gamma = nn.Parameter(torch.ones(num_freqs + 1))
        self.freq_beta = nn.Parameter(torch.zeros(num_freqs + 1))

        # Dynamic residual gate
        in_dim = 2 * (num_freqs + 1)
        self.content_gate_mlp = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, num_freqs + 1),
        )
        nn.init.zeros_(self.content_gate_mlp[-1].weight)
        nn.init.zeros_(self.content_gate_mlp[-1].bias)

    def _get_init_weights(self) -> torch.Tensor:
        w = torch.zeros(self.num_freqs + 1)
        w[0] = 3.5
        if self.num_freqs >= 1:
            w[1] = 1.0
        if self.num_freqs >= 2:
            w[2] = 0.5
        if self.num_freqs <= 4:
            if self.num_freqs >= 3:
                w[3:] = -2.0
        else:
            w[3:5] = -0.5
            w[5:] = -2.0
        return w

    def _build_scale_invariant_content(self, calibrated_spectrum: torch.Tensor) -> torch.Tensor:
        """Compute gate content features: [B, 2*(K+1)] = [density, ratio].

        DC uses learnable floor; AC uses fixed eps.
        """
        energy_signed = calibrated_spectrum.mean(dim=(1, 2))

        dc_floor = F.softplus(self.dc_floor_logit)
        dc_energy = energy_signed[:, 0:1].clamp_min(dc_floor)
        ac_energy = energy_signed[:, 1:].clamp_min(self.eps)
        energy = torch.cat([dc_energy, ac_energy], dim=-1)

        density = energy / energy.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        dc = energy[:, 0:1]
        ac = energy[:, 1:]
        ratio_ac = torch.log1p(ac / (dc + self.eps))
        ratio = torch.zeros_like(energy)
        ratio[:, 1:] = ratio_ac

        return torch.cat([density, ratio], dim=-1)

    def forward(self, x: torch.Tensor, mask=None, return_per_k: bool = False):
        """
        Args:
            x: [B, C, R, Phi]
            mask: [B, 1, R, Phi] or [B, R, Phi]
            return_per_k: also return calibrated_spectrum and weights

        Returns:
            ring_tokens: [B, R, C]
            (calibrated_spectrum, weights) if return_per_k
        """
        B, C, R, Phi = x.shape

        X = torch.fft.rfft(x, dim=-1, norm='forward')
        K = min(self.num_freqs, X.shape[-1] - 1)

        dc = X[..., 0].real.unsqueeze(-1)
        ac = torch.abs(X[..., 1:K + 1])
        mag = torch.cat([dc, ac], dim=-1)

        if K < self.num_freqs:
            mag = F.pad(mag, (0, self.num_freqs - K))

        gamma = self.freq_gamma.view(1, 1, 1, -1)
        beta = self.freq_beta.view(1, 1, 1, -1)
        calibrated_spectrum = gamma * mag + beta

        content = self._build_scale_invariant_content(calibrated_spectrum)
        delta = torch.tanh(self.content_gate_mlp(content))

        prior = self.prior_log_weights.view(1, -1)
        weights = torch.softmax(prior + delta, dim=-1)

        ring_tokens = (calibrated_spectrum * weights.view(B, 1, 1, -1)).sum(dim=-1)

        if mask is not None:
            mask = mask.to(device=x.device, dtype=x.dtype).clamp_min(0.0)
            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            if mask.shape[-2:] != (R, Phi):
                mask = F.adaptive_avg_pool2d(mask.unsqueeze(1), (R, Phi)).squeeze(1)
            ring_saliency = mask.mean(dim=-1).clamp_min(self.eps)
            ring_tokens = ring_tokens * ring_saliency.unsqueeze(1)

        ring_tokens = ring_tokens.permute(0, 2, 1).contiguous()

        if return_per_k:
            return ring_tokens, calibrated_spectrum, weights
        return ring_tokens


# =============================================================================
# Position Embedding
# =============================================================================

class PositionEmbeddingCoordsSine(nn.Module):
    """Sinusoidal position embedding for 1D coordinates."""

    def __init__(self, n_dim: int = 1, d_model: int = 64, temperature=10000, scale=None):
        super().__init__()
        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim
        self.scale = (scale if scale is not None else 1.0) * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)
        return F.pad(pos_emb, (0, self.padding))


# =============================================================================
# Ring Query Attention Blocks
# =============================================================================

class RingQueryAttentionBlock(nn.Module):
    """Single SRQA block: proposal query self-attention + ring cross-attention.

    need_weights=False when return_attention=False enables Flash/SDPA.
    """

    def __init__(self, in_dim, num_rqueries, num_pqueries, nheads=8):
        super().__init__()
        self.proposal_queries = nn.Parameter(torch.randn(1, num_pqueries, in_dim))
        self.query_self_attention = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_queries = nn.LayerNorm(in_dim)
        self.pos_embed = PositionEmbeddingCoordsSine(1, in_dim)
        self.ring_cross_attention = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(in_dim)

    def forward(self, x, ring_tokens=None, return_attention: bool = False):
        """
        Args:
            x: [B, C, R, Phi] (passed through unchanged)
            ring_tokens: [B, R, C] ring features from spectral decomposition
            return_attention: if True, return cross-attention weights [B, Q, R]

        Returns:
            x, out_tokens, ring_attention (ring_attention is None when return_attention=False)
        """
        B, C, R, Phi = x.shape
        queries = self.proposal_queries.repeat(B, 1, 1)

        queries = queries + self.query_self_attention(queries, queries, queries, need_weights=False)[0]
        queries = self.norm_queries(queries)

        if ring_tokens is None:
            ring_tokens = x.mean(dim=3).permute(0, 2, 1).contiguous()

        pe = self.pos_embed(torch.arange(ring_tokens.shape[1], device=ring_tokens.device).float().reshape(-1, 1))
        ring_tokens = ring_tokens + pe[None, ...]

        if return_attention:
            out_tokens, ring_attention = self.ring_cross_attention(queries, ring_tokens, ring_tokens, need_weights=True)
        else:
            out_tokens = self.ring_cross_attention(queries, ring_tokens, ring_tokens, need_weights=False)[0]
            ring_attention = None

        out_tokens = self.norm_out(out_tokens) + ring_tokens
        return x, out_tokens, ring_attention


class SpectralRingQueryAggregation(nn.Module):
    """Spectral Ring Query Aggregation head.

    Produces global descriptor [B, D] from BEV feature map [B, C, H, W].

    When use_sca=True, applies confidence-aware token modulation before
    ring query blocks:
      1. Spectral confidence: from per-ring frequency energy entropy
      2. Visibility confidence: ring-wise mean of visibility mask
      3. Joint confidence: c_joint = c_vis * c_spec
      4. Token modulation: ring_tokens = ring_tokens * (1 + lambda_c * c_joint)
    """

    def __init__(self, in_channels=1024, proj_channels=512, num_rqueries=32,
                 num_pqueries=32, num_layers=2, row_dim=32, num_freqs=8,
                 use_sca: bool = True, sca_lambda: float = 0.1):
        super().__init__()

        self.proj_c = nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = nn.BatchNorm2d(proj_channels)

        self.spectral_decomposition = SpectralRingDecomposition(num_freqs=num_freqs)

        # Confidence-aware token modulation config
        self.use_sca = use_sca
        self.sca_lambda = sca_lambda

        self.ring_query_blocks = nn.ModuleList([
            RingQueryAttentionBlock(proj_channels, num_rqueries, num_pqueries, nheads=8)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(num_layers * num_pqueries, row_dim)
        self.feature_scale = nn.Parameter(torch.tensor(18.0))

    # -----------------------------------------------------------------
    # Confidence helpers
    # -----------------------------------------------------------------

    def _compute_spectral_confidence(self, calibrated_spectrum: torch.Tensor,
                                     eps: float = 1e-7) -> torch.Tensor:
        """Per-ring spectral confidence from frequency energy distribution entropy.

        Args:
            calibrated_spectrum: [B, C, R, K+1] calibrated frequency magnitudes

        Returns:
            c_spec: [B, R, 1], values in [0, 1].
        """
        freq_energy = calibrated_spectrum.abs().mean(dim=1)
        freq_sum = freq_energy.sum(dim=-1, keepdim=True).clamp_min(eps)
        freq_dist = freq_energy / freq_sum
        freq_dist = freq_dist.clamp_min(eps)
        H = -(freq_dist * freq_dist.log()).sum(dim=-1)
        K_plus_1 = calibrated_spectrum.shape[-1]
        H_max = math.log(K_plus_1)
        c_spec = (1.0 - H / H_max).clamp(0.0, 1.0)
        return c_spec.unsqueeze(-1)

    def _compute_visibility_confidence(self, mask, R: int, Phi: int,
                                       eps: float = 1e-7):
        """Ring-wise visibility confidence from spatial mask.

        Returns: c_vis: [B, R, 1] or None
        """
        if mask is None:
            return None
        m = mask.to(dtype=torch.float32).clamp_min(0.0)
        if m.dim() == 4 and m.shape[1] == 1:
            m = m.squeeze(1)
        if m.shape[-2:] != (R, Phi):
            m = F.adaptive_avg_pool2d(
                m.unsqueeze(1), (R, Phi)).squeeze(1)
        c_vis = m.mean(dim=-1).clamp(0.0, 1.0)
        return c_vis.unsqueeze(-1)

    def _apply_confidence_modulation(self, ring_tokens: torch.Tensor,
                                     c_joint: torch.Tensor) -> torch.Tensor:
        """Apply confidence-aware residual scaling to ring tokens.

        Args:
            ring_tokens: [B, R, C]
            c_joint: [B, R, 1]

        Returns:
            modulated ring tokens: [B, R, C]
        """
        return ring_tokens * (1.0 + self.sca_lambda * c_joint)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, x, mask=None, return_per_k: bool = False,
                return_attention: bool = False):
        """
        Args:
            x: [B, C, H, W]
            mask: visibility mask for spectral decomposition
            return_per_k: return decomposed ring tokens for ring alignment loss
            return_attention: return cross-attention weights for ring saliency

        Returns:
            out: [B, D] global descriptor (always)
            attns: list of [B, Q, R] or None
            ring_tokens_raw: [B, R, C] if return_per_k
        """
        x_proj = self.norm_input(self.proj_c(x))
        B, C, R, Phi = x_proj.shape

        # --- Spectral Decomposition + Confidence Modulation ---
        if self.use_sca:
            ring_tokens, calibrated_spectrum, _weights = self.spectral_decomposition(
                x_proj, mask, return_per_k=True)
            ring_tokens_raw = ring_tokens

            c_spec = self._compute_spectral_confidence(calibrated_spectrum)
            c_vis = self._compute_visibility_confidence(mask, R, Phi)

            if c_vis is not None:
                c_joint = c_vis * c_spec
            else:
                c_joint = c_spec

            ring_tokens = self._apply_confidence_modulation(ring_tokens, c_joint)
            del calibrated_spectrum, _weights
        else:
            if return_per_k:
                ring_tokens, _mag, _w = self.spectral_decomposition(x_proj, mask, return_per_k=True)
                ring_tokens_raw = ring_tokens
                del _mag, _w
            else:
                ring_tokens = self.spectral_decomposition(x_proj, mask, return_per_k=False)
                ring_tokens_raw = ring_tokens
            ring_tokens = ring_tokens

        ring_tokens = ring_tokens * self.feature_scale

        # --- Ring Query Blocks ---
        outs = []
        attns = [] if return_attention else None
        for block in self.ring_query_blocks:
            x_proj, out, attn = block(x_proj, ring_tokens=ring_tokens, return_attention=return_attention)
            outs.append(out)
            if return_attention:
                attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1)).flatten(1)

        if return_per_k:
            return out, attns, ring_tokens_raw
        return out, attns