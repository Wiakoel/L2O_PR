"""
network/Node_Prompting.py — Anchor-Guided Saliency Prompting (AGSP, §4.2)

Paper reference: Structural Fingerprint-Aware Attention (SFAA).

This module constructs saliency prompts from structural fingerprint element
candidates (anchors). Five semantic categories serve as structural anchors:
  poles, traffic signs, tree trunks, building corners, road forks.

Components:
  A. Fingerprint Saliency Field Construction (Eq. 7)
     - Per-type Gaussian saliency field from sparse structural anchors
     - OSM: Cartesian coordinate system, per-type [B, K, H, W]
     - PC:  Polar coordinate system, per-type [B, K, R, Phi]
     - Both use scatter_reduce_ with 'amax' for deterministic max-merge.

  B. Relation-Biased Structural Attention (Eq. 8)
     - Anchor token sampling from post-modulation x5 feature map
     - Token = center_feat + r_normalized
     - Local attention with geometric relation bias (d_ij, Δρ, Δθ, same_type)
     - Bounded difference residual injection
     - Gated auxiliary path complementing main saliency modulation

RESEARCH CORE — do not modify scatter/augmentation/coordinate semantics.
"""

import math
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Semantic Constants (single source of truth)
# =============================================================================

NODE_SEMANTIC_TYPES = ['road_fork', 'building_corner', 'trunk', 'pole', 'traffic_sign']
NUM_NODE_TYPES = len(NODE_SEMANTIC_TYPES)
TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_SEMANTIC_TYPES)}


# =============================================================================
# A. Fingerprint Saliency Field Construction (Eq. 7)
# =============================================================================

def _generate_gaussian_kernel(radius: int, sigma: float,
                              device: torch.device) -> torch.Tensor:
    """Generate truncated 2D isotropic Gaussian kernel, peak normalized to 1.0.

    Returns: [2*radius+1, 2*radius+1]
    """
    radius = int(radius)
    sigma = float(max(sigma, 1e-6))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return kernel / kernel.max().clamp_min(1e-6)


# -----------------------------------------------------------------------------
# OSM Cartesian Saliency Fields
# -----------------------------------------------------------------------------

def build_osm_saliency_fields(
    node_coords_list: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    aug_params_list: List[Dict],
    H: int, W: int,
    device: torch.device,
    tile_size: float = 100.0,
    dilation_radius: float = 3.0,
    **_kwargs,
) -> torch.Tensor:
    """Generate per-type Gaussian saliency fields in Cartesian coordinates.

    Args:
        node_coords_list: [B] list of {type_name: [N, 2] tensor} or tensor
        aug_params_list: [B] list of {rot_k, flip_x, flip_y}
        H, W: output spatial dimensions
        tile_size: OSM tile size in meters

    Returns:
        saliency_fields: [B, NUM_NODE_TYPES, H, W], values in [0, 1]
    """
    B = len(node_coords_list)
    res = tile_size / H
    cx, cy = W / 2.0, H / 2.0

    r = int(dilation_radius)
    sigma = max(float(dilation_radius) / 2.0, 1.0)
    g_kernel = _generate_gaussian_kernel(r, sigma, device=device)

    saliency_fields = torch.zeros(B, NUM_NODE_TYPES, H, W, dtype=torch.float32, device=device)

    for b in range(B):
        item = node_coords_list[b]
        aug = aug_params_list[b]
        rot_k = aug.get('rot_k', 0)
        flip_x = aug.get('flip_x', False)
        flip_y = aug.get('flip_y', False)

        if isinstance(item, dict):
            for type_name, coords in item.items():
                if type_name not in TYPE_TO_IDX:
                    continue
                type_idx = TYPE_TO_IDX[type_name]
                coords = coords.to(device=device, dtype=torch.float32)
                if coords.numel() == 0:
                    continue
                _write_cartesian_saliency(
                    saliency_fields[b, type_idx], coords,
                    cx, cy, res, H, W, rot_k, flip_x, flip_y, g_kernel)
        else:
            coords = item.to(device=device, dtype=torch.float32)
            if coords.numel() > 0:
                _write_cartesian_saliency(
                    saliency_fields[b, 0], coords,
                    cx, cy, res, H, W, rot_k, flip_x, flip_y, g_kernel)

    return saliency_fields


def _write_cartesian_saliency(
    mask_b: torch.Tensor, coords: torch.Tensor,
    cx: float, cy: float, res: float, H: int, W: int,
    rot_k: int, flip_x: bool, flip_y: bool,
    g_kernel: torch.Tensor,
):
    """Vectorized Gaussian writer in Cartesian space. Modifies mask_b in-place.

    Uses scatter_reduce_ with 'amax' (deterministic).
    """
    cols = coords[:, 0] / res + cx
    rows = -coords[:, 1] / res + cy

    # Augmentation sync
    if rot_k > 0:
        cols_c, rows_c = cols - cx, rows - cy
        for _ in range(rot_k):
            cols_c, rows_c = rows_c, -cols_c
        cols, rows = cols_c + cx, rows_c + cy
    if flip_x:
        cols = W - 1 - cols
    if flip_y:
        rows = H - 1 - rows

    rows_i = torch.round(rows).long()
    cols_i = torch.round(cols).long()

    valid = (rows_i >= 0) & (rows_i < H) & (cols_i >= 0) & (cols_i < W)
    if not valid.any():
        return
    rows_i, cols_i = rows_i[valid], cols_i[valid]
    N = rows_i.shape[0]

    kH, kW = g_kernel.shape
    r_h, r_w = kH // 2, kW // 2
    dy = torch.arange(-r_h, r_h + 1, device=mask_b.device)
    dx = torch.arange(-r_w, r_w + 1, device=mask_b.device)

    all_rows = (rows_i.view(N, 1, 1) + dy.view(1, kH, 1)).expand(N, kH, kW)
    all_cols = (cols_i.view(N, 1, 1) + dx.view(1, 1, kW)).expand(N, kH, kW)
    all_vals = g_kernel.unsqueeze(0).expand(N, kH, kW)

    in_bounds = (all_rows >= 0) & (all_rows < H) & (all_cols >= 0) & (all_cols < W)
    flat_idx = all_rows[in_bounds] * W + all_cols[in_bounds]
    mask_b.view(-1).scatter_reduce_(0, flat_idx, all_vals[in_bounds],
                                     reduce='amax', include_self=True)


# -----------------------------------------------------------------------------
# PC Polar Saliency Fields
# -----------------------------------------------------------------------------

def build_pc_saliency_fields(
    pc_node_coords_list: List[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    pc_aug_params_list: List[Dict],
    R: int = 480, Phi: int = 360,
    device: torch.device = None,
    r_min: float = 3.0, r_max: float = 50.0,
    dilation_r: int = 2, dilation_phi: int = 3,
    **_kwargs,
) -> torch.Tensor:
    """Generate per-type Gaussian saliency fields in polar coordinates.

    Returns: [B, NUM_NODE_TYPES, R, Phi], values in [0, 1].
    Uses anisotropic Gaussian kernel. Phi dimension wraps circularly.
    """
    B = len(pc_node_coords_list)
    saliency_fields = torch.zeros(B, NUM_NODE_TYPES, R, Phi, dtype=torch.float32, device=device)

    # Anisotropic Gaussian kernel in polar bins
    dr = torch.arange(-dilation_r, dilation_r + 1, device=device, dtype=torch.float32)
    dphi = torch.arange(-dilation_phi, dilation_phi + 1, device=device, dtype=torch.float32)
    dr_grid, dphi_grid = torch.meshgrid(dr, dphi, indexing='ij')
    sigma_r = max(float(dilation_r) / 2.0, 0.75)
    sigma_phi = max(float(dilation_phi) / 2.0, 1.0)
    g_kernel_polar = torch.exp(
        -0.5 * ((dr_grid / sigma_r) ** 2 + (dphi_grid / sigma_phi) ** 2))
    g_kernel_polar = g_kernel_polar / g_kernel_polar.max().clamp_min(1e-6)

    for b in range(B):
        item = pc_node_coords_list[b]
        theta = pc_aug_params_list[b].get('theta', 0.0)

        if isinstance(item, dict):
            for type_name, coords in item.items():
                if type_name not in TYPE_TO_IDX:
                    continue
                type_idx = TYPE_TO_IDX[type_name]
                coords = coords.to(device=device, dtype=torch.float32)
                if coords.numel() == 0:
                    continue
                _write_polar_saliency(
                    saliency_fields[b, type_idx], coords, theta,
                    R, Phi, r_min, r_max, g_kernel_polar)
        else:
            coords = item.to(device=device, dtype=torch.float32)
            if coords.numel() > 0:
                _write_polar_saliency(
                    saliency_fields[b, 0], coords, theta,
                    R, Phi, r_min, r_max, g_kernel_polar)

    return saliency_fields


def _write_polar_saliency(
    mask_b: torch.Tensor,
    coords: torch.Tensor,
    theta: float,
    R: int, Phi: int,
    r_min: float, r_max: float,
    g_kernel_polar: torch.Tensor,
):
    """Vectorized Gaussian writer in polar space. Modifies mask_b in-place.

    Circular phi wrapping via ``% Phi``. scatter_reduce_ with 'amax'.
    """
    x_all, y_all = coords[:, 0], coords[:, 1]
    rho = torch.sqrt(x_all ** 2 + y_all ** 2)
    phi = torch.atan2(y_all, x_all) + float(theta)
    phi = torch.remainder(phi + np.pi, 2 * np.pi) - np.pi

    r_idx = torch.floor((rho - r_min) / (r_max - r_min) * (R - 1)).long()
    phi_idx = torch.floor((phi + np.pi) / (2 * np.pi) * Phi).long() % Phi

    valid = (r_idx >= 0) & (r_idx < R)
    if not valid.any():
        return
    r_idx, phi_idx = r_idx[valid], phi_idx[valid]
    N = r_idx.shape[0]

    kR, kP = g_kernel_polar.shape
    rr, rp = kR // 2, kP // 2
    dr = torch.arange(-rr, rr + 1, device=mask_b.device)
    dp = torch.arange(-rp, rp + 1, device=mask_b.device)

    all_r = (r_idx.view(N, 1, 1) + dr.view(1, kR, 1)).expand(N, kR, kP)
    all_phi = ((phi_idx.view(N, 1, 1) + dp.view(1, 1, kP)) % Phi).expand(N, kR, kP)
    all_vals = g_kernel_polar.unsqueeze(0).expand(N, kR, kP)

    r_valid = (all_r >= 0) & (all_r < R)
    flat_idx = all_r[r_valid] * Phi + all_phi[r_valid]
    mask_b.view(-1).scatter_reduce_(0, flat_idx, all_vals[r_valid],
                                     reduce='amax', include_self=True)


# =============================================================================
# B. Relation-Biased Structural Attention (Eq. 8)
# =============================================================================

class RelationBiasedLocalAttention(nn.Module):
    """Relation-biased structural attention for anchor-guided saliency prompting.

    Implements Eq. 8: attention with geometric relation bias ψ(r_ij) among
    neighboring structural fingerprint element candidates (anchors).

    Operates on x5 feature map [B, C, H5, W5] (polar space, typically 30x22).
    Samples anchor tokens from the POST-MODULATION feature map → local attention
    with relation bias → difference residual → bounded scatter-back → gated addition.

    Token construction: center_feat (bilinear sample at node position) + r_normalized.

    Design constraints (do not change):
      - Operates only at x5 level, reads POST-BOOST features
      - Difference residual only: delta = tanh(W_residual(z' - z))
      - Bounded gate: lambda = sigmoid(raw) * lambda_max
      - Degrades to zero residual when lambda~0, W_residual~0

    Args:
        x5_channels: channel dim of x5 (512)
        hidden_dim: internal token dim (64)
        num_node_types: for same_type relation feature
        neighbor_radius_m: neighbor threshold in meters
        K_safe: safety cap on neighbor count (OOM prevention only)
        scatter_radius: Gaussian scatter kernel radius in x5 pixels
        lambda_init: initial gate value
        lambda_max: hard upper bound on gate
    """

    def __init__(self, x5_channels: int = 512, hidden_dim: int = 64,
                 num_node_types: int = 5, neighbor_radius_m: float = 5.0,
                 K_safe: int = 48, scatter_radius: int = 1,
                 lambda_init: float = 0.005, lambda_max: float = 0.03):
        super().__init__()
        self.x5_channels = x5_channels
        self.hidden_dim = hidden_dim
        self.neighbor_radius_m = neighbor_radius_m
        self.K_safe = K_safe
        self.scatter_radius = scatter_radius
        self.lambda_max = lambda_max

        # Token construction: center_feat + r_normalized
        token_input_dim = x5_channels + 1
        self.input_proj = nn.Sequential(
            nn.Linear(token_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Relation-biased local attention (single layer)
        # Relation features: [d_ij, delta_r_ij, cos(delta_theta_ij), sin(delta_theta_ij), same_type]
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        # Difference residual projection — ZERO-INITIALIZED for safe startup
        self.W_residual = nn.Linear(hidden_dim, x5_channels, bias=False)
        nn.init.zeros_(self.W_residual.weight)

        # Learnable gate with hard upper bound
        effective_init = lambda_init
        if effective_init >= lambda_max:
            effective_init = lambda_max * 0.5
        target_sigmoid = min(max(effective_init / lambda_max, 0.01), 0.99)
        raw = math.log(target_sigmoid / (1.0 - target_sigmoid))
        self.attention_gate_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

        # Scatter Gaussian kernel (fixed, small)
        r = self.scatter_radius
        coords = torch.arange(-r, r + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        sigma = max(r / 2.0, 0.5)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.max().clamp_min(1e-6)
        self.register_buffer('scatter_kernel', kernel)

        # Debug stats (cached for logging, no gradient)
        self._debug_stats = {
            'lambda_val': 0.0,
            'delta_abs_mean': 0.0,
            'avg_num_nodes': 0.0,
            'avg_num_neighbors': 0.0,
        }

    def get_attention_gate(self) -> torch.Tensor:
        """Bounded gate: lambda in [0, lambda_max]."""
        return torch.sigmoid(self.attention_gate_raw) * self.lambda_max

    def forward(self, x5: torch.Tensor, prompting_info: dict) -> torch.Tensor:
        """Compute gated prompting residual for x5.

        Args:
            x5: [B, C, H5, W5] feature map (POST-BOOST, after saliency modulation)
            prompting_info: dict with keys:
                'positions_x5': list of B tensors, each [N_b, 2] float (r, phi in x5 grid)
                'coords_orig':  list of B tensors, each [N_b, 2] float (original meters)
                'type_ids':     list of B tensors, each [N_b] long
                'r_normalized': list of B tensors, each [N_b] float in [0, 1]

        Returns:
            prompting_residual: [B, C, H5, W5], the gated residual to ADD to x5.
        """
        B, C, H5, W5 = x5.shape
        device = x5.device
        dtype = x5.dtype

        attention_gate = self.get_attention_gate()

        total_nodes = 0
        total_neighbors = 0
        n_samples_with_nodes = 0

        sample_deltas = []

        for b in range(B):
            pos_x5_b = prompting_info['positions_x5'][b].to(device=device, dtype=dtype)
            coords_b = prompting_info['coords_orig'][b].to(device=device, dtype=dtype)
            type_ids_b = prompting_info['type_ids'][b].to(device=device)
            r_norm_b = prompting_info['r_normalized'][b].to(device=device, dtype=dtype)

            N = pos_x5_b.shape[0]
            if N == 0:
                sample_deltas.append(
                    torch.zeros(C, H5, W5, device=device, dtype=dtype))
                continue

            # 1. Sample node tokens from x5 via bilinear interpolation
            r_coords = pos_x5_b[:, 0]
            phi_coords = pos_x5_b[:, 1]

            grid_h = r_coords / max(H5 - 1, 1) * 2.0 - 1.0
            grid_w = phi_coords / max(W5 - 1, 1) * 2.0 - 1.0
            grid = torch.stack([grid_w, grid_h], dim=-1).view(1, N, 1, 2)
            grid = grid.clamp(-1.0, 1.0)

            center_feat = F.grid_sample(
                x5[b:b + 1], grid, mode='bilinear',
                align_corners=True, padding_mode='border'
            ).squeeze(0).squeeze(-1).permute(1, 0)  # [N, C]

            # Assemble token input: center_feat + r_normalized
            token_input = torch.cat([
                center_feat,
                r_norm_b.unsqueeze(-1),
            ], dim=-1)  # [N, C+1]

            z = self.input_proj(token_input)

            # 2. Local attention with relation bias
            if N == 1:
                z_prime = z
            else:
                z_prime = self._relation_biased_attention(z, coords_b, type_ids_b, r_norm_b)

            # 3. Difference residual
            diff = z_prime - z
            residual_per_node = torch.tanh(self.W_residual(diff))

            # 4. Scatter residual back to x5 with small Gaussian
            sample_delta = self._build_saliency_residual_map(
                residual_per_node, pos_x5_b, C, H5, W5)
            sample_deltas.append(sample_delta)

            total_nodes += N
            n_samples_with_nodes += 1

        residual_map = torch.stack(sample_deltas, dim=0)
        residual_map = torch.tanh(residual_map)

        prompting_residual = attention_gate * residual_map

        with torch.no_grad():
            self._debug_stats['lambda_val'] = attention_gate.item()
            self._debug_stats['delta_abs_mean'] = residual_map.abs().mean().item()
            self._debug_stats['avg_num_nodes'] = (
                total_nodes / max(n_samples_with_nodes, 1))

        return prompting_residual

    def _relation_biased_attention(self, z: torch.Tensor, coords: torch.Tensor,
                                    type_ids: torch.Tensor, r_norm: torch.Tensor
                                    ) -> torch.Tensor:
        """Single-layer relation-biased local attention among nodes.

        Relation features are rotation-robust: d_ij, delta_r, cos/sin(delta_theta), same_type.

        Args:
            z: [N, hidden_dim] node tokens
            coords: [N, 2] original coordinates in meters
            type_ids: [N] node type indices
            r_norm: [N] normalized radial positions

        Returns:
            z_prime: [N, hidden_dim] updated tokens (z + attention residual)
        """
        N = z.shape[0]
        device = z.device

        x_all, y_all = coords[:, 0], coords[:, 1]
        r_all = torch.sqrt(x_all ** 2 + y_all ** 2)
        theta_all = torch.atan2(y_all, x_all)

        dx = x_all.unsqueeze(1) - x_all.unsqueeze(0)
        dy = y_all.unsqueeze(1) - y_all.unsqueeze(0)
        d_ij = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        delta_r = r_all.unsqueeze(1) - r_all.unsqueeze(0)
        delta_theta = theta_all.unsqueeze(1) - theta_all.unsqueeze(0)
        delta_theta = torch.remainder(delta_theta + math.pi, 2 * math.pi) - math.pi

        same_type = (type_ids.unsqueeze(1) == type_ids.unsqueeze(0)).float()

        neighbor_mask = (d_ij < self.neighbor_radius_m)
        neighbor_mask.fill_diagonal_(True)

        # K_safe: OOM prevention only
        neighbor_count = neighbor_mask.sum(dim=1)
        if neighbor_count.max().item() > self.K_safe:
            for i in range(N):
                if neighbor_count[i] > self.K_safe:
                    dists_i = d_ij[i]
                    dists_i[~neighbor_mask[i]] = float('inf')
                    _, topk_idx = dists_i.topk(self.K_safe, largest=False)
                    new_mask = torch.zeros(N, dtype=torch.bool, device=device)
                    new_mask[topk_idx] = True
                    neighbor_mask[i] = new_mask

        # Relation bias
        d_ij_norm = d_ij / max(self.neighbor_radius_m, 1.0)
        delta_r_norm = delta_r / max(self.neighbor_radius_m, 1.0)
        rel_feat = torch.stack([
            d_ij_norm,
            delta_r_norm,
            torch.cos(delta_theta),
            torch.sin(delta_theta),
            same_type,
        ], dim=-1)

        relation_bias = self.relation_mlp(rel_feat).squeeze(-1)

        # Attention
        q = self.Wq(z)
        k = self.Wk(z)
        v = self.Wv(z)

        scale = math.sqrt(self.hidden_dim)
        attn_logits = (q @ k.T) / scale + relation_bias
        attn_logits = attn_logits.masked_fill(~neighbor_mask, float('-inf'))

        valid_rows = neighbor_mask.any(dim=1)
        attn_weights = torch.zeros_like(attn_logits)
        if valid_rows.any():
            attn_weights[valid_rows] = torch.softmax(
                attn_logits[valid_rows], dim=-1)

        # Residual update: only nodes with real neighbors get residual
        real_neighbor_count = neighbor_mask.float().sum(dim=1) - 1.0
        has_real_neighbors = (real_neighbor_count > 0).float().unsqueeze(1)

        attended = attn_weights @ v
        z_prime = z + attended * has_real_neighbors

        with torch.no_grad():
            avg_real_neighbors = real_neighbor_count.mean().item()
            self._debug_stats['avg_num_neighbors'] = avg_real_neighbors

        return z_prime

    def _build_saliency_residual_map(self, residual_per_node: torch.Tensor,
                                      positions_x5: torch.Tensor,
                                      C: int, H5: int, W5: int) -> torch.Tensor:
        """Build residual feature map for one sample via vectorized scatter_add.

        NOT in-place. Phi dimension (W) wraps circularly.

        Args:
            residual_per_node: [N, C] bounded residual per node
            positions_x5: [N, 2] (r, phi) float positions in x5 grid
            C, H5, W5: target map dimensions

        Returns:
            residual_map: [C, H5, W5]
        """
        N = residual_per_node.shape[0]
        device = residual_per_node.device
        kernel = self.scatter_kernel
        kH, kW = kernel.shape
        r_h, r_w = kH // 2, kW // 2

        rows_i = torch.round(positions_x5[:, 0]).long()
        cols_i = torch.round(positions_x5[:, 1]).long()

        dy = torch.arange(-r_h, r_h + 1, device=device)
        dx = torch.arange(-r_w, r_w + 1, device=device)

        all_r = (rows_i.view(N, 1, 1) + dy.view(1, kH, 1)).expand(N, kH, kW)
        all_c = ((cols_i.view(N, 1, 1) + dx.view(1, 1, kW)) % W5).expand(N, kH, kW)

        valid = (all_r >= 0) & (all_r < H5)

        all_w = kernel.unsqueeze(0).expand(N, kH, kW)
        node_idx = torch.arange(N, device=device).view(N, 1, 1).expand(N, kH, kW)

        flat_idx = (all_r[valid] * W5 + all_c[valid]).long()
        weights_v = all_w[valid]
        node_idx_v = node_idx[valid]

        weighted_delta = weights_v.unsqueeze(1) * residual_per_node[node_idx_v]

        result = torch.zeros(C, H5 * W5, device=device, dtype=residual_per_node.dtype)
        idx_expanded = flat_idx.unsqueeze(0).expand(C, -1)
        result.scatter_add_(1, idx_expanded, weighted_delta.permute(1, 0))

        return result.view(C, H5, W5)

    def log_prompting_stats(self) -> dict:
        """Return current prompting statistics for logging."""
        return dict(self._debug_stats)


# =============================================================================
# C. Prompting Info Builders (coordinate projection for attention module)
# =============================================================================

def build_osm_prompting_info(
    node_coords_list: list,
    aug_params_list: list,
    map_H: int, map_W: int,
    tile_size: float,
    polar_R: int, polar_Phi: int,
    x5_H: int, x5_W: int,
    device: torch.device,
) -> dict:
    """Build prompting_info for OSM branch.

    Converts node coordinates from Cartesian meters to x5 polar grid positions,
    applying the same augmentation (rot_k, flip_x, flip_y) as the saliency field
    pipeline to ensure spatial alignment.

    OSM pipeline coordinate chain:
      meters -> Cartesian pixels -> augmentation -> polar (480x360) -> x5 (30x22)
    """
    B = len(node_coords_list)

    res = tile_size / map_H
    cx, cy = map_W / 2.0, map_H / 2.0
    center_x = map_W // 2
    center_y = map_H // 2

    all_pos_x5 = []
    all_coords_orig = []
    all_type_ids = []
    all_r_norm = []

    for b in range(B):
        item = node_coords_list[b]
        aug = aug_params_list[b]
        rot_k = aug.get('rot_k', 0)
        flip_x = aug.get('flip_x', False)
        flip_y = aug.get('flip_y', False)

        pos_x5_list = []
        coords_orig_list = []
        type_id_list = []
        r_norm_list = []

        if isinstance(item, dict):
            for type_name, coords in item.items():
                if type_name not in TYPE_TO_IDX:
                    continue
                type_idx = TYPE_TO_IDX[type_name]
                coords = coords.to(dtype=torch.float32)
                if coords.numel() == 0:
                    continue

                coords_orig_list.append(coords)

                # Cartesian pixel coordinates (consistent with saliency field)
                cols = coords[:, 0] / res + cx
                rows = -coords[:, 1] / res + cy

                # Augmentation sync (identical to _write_cartesian_saliency)
                if rot_k > 0:
                    cols_c, rows_c = cols - cx, rows - cy
                    for _ in range(rot_k):
                        cols_c, rows_c = rows_c, -cols_c
                    cols, rows = cols_c + cx, rows_c + cy
                if flip_x:
                    cols = map_W - 1 - cols
                if flip_y:
                    rows = map_H - 1 - rows

                # Cartesian pixel -> polar coordinates
                px_dx = cols - center_x
                px_dy = center_y - rows
                rho = torch.sqrt(px_dx ** 2 + px_dy ** 2 + 1e-8)
                theta = torch.atan2(px_dy, px_dx)

                r_polar = rho / max(center_x, 1) * (polar_R - 1)
                phi_polar = (theta + math.pi) / (2 * math.pi) * (polar_Phi - 1)

                r_x5 = r_polar * (x5_H / polar_R)
                phi_x5 = phi_polar * (x5_W / polar_Phi)

                valid = (r_x5 >= 0) & (r_x5 < x5_H) & (phi_x5 >= 0) & (phi_x5 < x5_W)

                if valid.any():
                    pos_x5_list.append(torch.stack([r_x5[valid], phi_x5[valid]], dim=1))
                    coords_orig_list[-1] = coords[valid]
                    type_id_list.append(
                        torch.full((valid.sum().item(),), type_idx, dtype=torch.long))
                    r_norm_list.append(
                        (rho[valid] / max(center_x, 1)).clamp(0, 1))
                else:
                    coords_orig_list.pop()

        if pos_x5_list:
            all_pos_x5.append(torch.cat(pos_x5_list, dim=0))
            all_coords_orig.append(torch.cat(coords_orig_list, dim=0))
            all_type_ids.append(torch.cat(type_id_list, dim=0))
            all_r_norm.append(torch.cat(r_norm_list, dim=0))
        else:
            all_pos_x5.append(torch.zeros(0, 2, dtype=torch.float32))
            all_coords_orig.append(torch.zeros(0, 2, dtype=torch.float32))
            all_type_ids.append(torch.zeros(0, dtype=torch.long))
            all_r_norm.append(torch.zeros(0, dtype=torch.float32))

    return {
        'positions_x5': all_pos_x5,
        'coords_orig': all_coords_orig,
        'type_ids': all_type_ids,
        'r_normalized': all_r_norm,
    }


def build_pc_prompting_info(
    pc_node_coords_list: list,
    pc_aug_params_list: list,
    R: int, Phi: int,
    r_min: float, r_max: float,
    x5_H: int, x5_W: int,
    device: torch.device,
) -> dict:
    """Build prompting_info for PC branch.

    Converts node coordinates from LiDAR local frame to x5 polar grid positions,
    applying theta rotation augmentation consistent with build_pc_saliency_fields.

    PC pipeline coordinate chain:
      LiDAR meters (x,y) -> polar (rho, phi) + theta aug -> grid (R, Phi) -> x5
    """
    B = len(pc_node_coords_list)

    all_pos_x5 = []
    all_coords_orig = []
    all_type_ids = []
    all_r_norm = []

    for b in range(B):
        item = pc_node_coords_list[b]
        theta = pc_aug_params_list[b].get('theta', 0.0)

        pos_x5_list = []
        coords_orig_list = []
        type_id_list = []
        r_norm_list = []

        if isinstance(item, dict):
            for type_name, coords in item.items():
                if type_name not in TYPE_TO_IDX:
                    continue
                type_idx = TYPE_TO_IDX[type_name]
                coords = coords.to(dtype=torch.float32)
                if coords.numel() == 0:
                    continue

                coords_orig_list.append(coords)

                x_all, y_all = coords[:, 0], coords[:, 1]
                rho = torch.sqrt(x_all ** 2 + y_all ** 2 + 1e-8)
                phi = torch.atan2(y_all, x_all) + float(theta)
                phi = torch.remainder(phi + math.pi, 2 * math.pi) - math.pi

                r_idx = torch.floor(
                    (rho - r_min) / max(r_max - r_min, 1e-6) * (R - 1)).float()
                phi_idx = (torch.floor(
                    (phi + math.pi) / (2 * math.pi) * Phi).long() % Phi).float()

                r_x5 = r_idx * (x5_H / R)
                phi_x5 = phi_idx * (x5_W / Phi)

                valid = (r_x5 >= 0) & (r_x5 < x5_H)

                if valid.any():
                    phi_x5_valid = phi_x5[valid].clamp(0, x5_W - 1e-3)
                    pos_x5_list.append(
                        torch.stack([r_x5[valid], phi_x5_valid], dim=1))
                    coords_orig_list[-1] = coords[valid]
                    type_id_list.append(
                        torch.full((valid.sum().item(),), type_idx, dtype=torch.long))
                    r_norm_list.append(
                        ((rho[valid] - r_min) / max(r_max - r_min, 1e-6)).clamp(0, 1))
                else:
                    coords_orig_list.pop()

        if pos_x5_list:
            all_pos_x5.append(torch.cat(pos_x5_list, dim=0))
            all_coords_orig.append(torch.cat(coords_orig_list, dim=0))
            all_type_ids.append(torch.cat(type_id_list, dim=0))
            all_r_norm.append(torch.cat(r_norm_list, dim=0))
        else:
            all_pos_x5.append(torch.zeros(0, 2, dtype=torch.float32))
            all_coords_orig.append(torch.zeros(0, 2, dtype=torch.float32))
            all_type_ids.append(torch.zeros(0, dtype=torch.long))
            all_r_norm.append(torch.zeros(0, dtype=torch.float32))

    return {
        'positions_x5': all_pos_x5,
        'coords_orig': all_coords_orig,
        'type_ids': all_type_ids,
        'r_normalized': all_r_norm,
    }