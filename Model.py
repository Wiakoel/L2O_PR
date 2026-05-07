"""
network/Model.py — SFPR Top-Level Cross-Modal Model

Paper reference: §4 Methodology — full pipeline.

Components:
  - OSMEncoderWrapper: OSM raster → embedding → polar → saliency modulation → BEV encoder
  - SFPRModel: dual-branch (OSM + PC) model with structural fingerprint-aware attention

SFPRModel contains:
  1. OSM branch (with physical visibility mask and saliency modulation)
  2. LiDAR/point-cloud branch
  3. Structural fingerprint-aware attention prompts (AGSP, §4.2)
  4. Shared global descriptor generation (FAFA, §4.2)
  5. Optional return of fingerprint-aware attention for fingerprint extraction/matching (§4.3)

Forward contract:
    Base:       (osm_desc, pc_desc) both [B, D] L2-normalized
    +per_k:     appends (pc_pooled, osm_pooled)
    +attention: appends (pc_attn, osm_attn) — fingerprint-aware attention maps

RESEARCH CORE — do not modify:
  - Per-type per-channel boost embedding structure
  - Saliency modulation (einsum saliency_field × saliency_boost)
  - Physical visibility mask pipeline
  - Polar grid cache keying
  - Forward input/output contract
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.Encoder import BEVUNetEncoder
from network.Encoder import MapEncoder
from network.Encoder import PointCloudBEVEncoder
from network.Node_Prompting import (
    build_osm_saliency_fields, build_pc_saliency_fields,
    NODE_SEMANTIC_TYPES, NUM_NODE_TYPES,
    RelationBiasedLocalAttention, build_osm_prompting_info, build_pc_prompting_info,
)

logger = logging.getLogger(__name__)


# =============================================================================
# OSM Encoder Wrapper
# =============================================================================

class OSMEncoderWrapper(nn.Module):
    """OSM Encoder with polar transform, physical visibility mask, and saliency modulation.

    Pipeline:
      1. MapEncoder: discrete raster → continuous embeddings [B, 48, H, W]
      2. Cartesian → Polar via cached grid_sample
      3. Physical visibility mask from ray-casting occluders
      4. Saliency modulation: per-type saliency_field x saliency_boost -> spatial-channel modulation at x5
      5. BEVUNetEncoder -> descriptor
    """

    def __init__(self, conf, BEV_net=None, use_osm_saliency: bool = False,
                 dilation_radius: float = 3.0, use_physical_visibility: bool = True):
        super().__init__()
        self.map_encoder = MapEncoder(conf['model']['map_encoder'])
        self.radial_resolution = 480
        self.angular_resolution = 360
        self.BEV_model = BEV_net
        self.use_osm_saliency = use_osm_saliency
        self.dilation_radius = dilation_radius
        self.use_physical_visibility = use_physical_visibility
        self._grid_cache = {}

    def _get_polar_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Lazy-cached polar sampling grid. Keyed by (H, W), auto-refreshes on device change."""
        cache_key = (H, W)
        if cache_key in self._grid_cache:
            cached = self._grid_cache[cache_key]
            if cached.device == device:
                return cached

        center_x, center_y = W // 2, H // 2
        theta = torch.linspace(-np.pi, np.pi, self.angular_resolution, device=device)
        radius = torch.linspace(0, center_x, self.radial_resolution, device=device)
        grid_x = center_x + radius.view(-1, 1) * torch.cos(theta).view(1, -1)
        grid_y = center_y - radius.view(-1, 1) * torch.sin(theta).view(1, -1)
        grid_x = (grid_x / (W - 1) * 2 - 1).view(1, self.radial_resolution, self.angular_resolution, 1)
        grid_y = (grid_y / (H - 1) * 2 - 1).view(1, self.radial_resolution, self.angular_resolution, 1)
        grid = torch.cat((grid_x, grid_y), dim=-1)

        self._grid_cache[cache_key] = grid
        return grid

    def cartesian_to_polar(self, map_data):
        B, C, H, W = map_data.shape
        grid = self._get_polar_grid(H, W, map_data.device).expand(B, -1, -1, -1)
        return F.grid_sample(map_data, grid, mode='bilinear', align_corners=True)

    def cartesian_to_polar_mask(self, mask_cart: torch.Tensor) -> torch.Tensor:
        B, _, H, W = mask_cart.shape
        grid = self._get_polar_grid(H, W, mask_cart.device).expand(B, -1, -1, -1)
        return F.grid_sample(mask_cart, grid, mode='bilinear', align_corners=True)

    def generate_physical_visibility_mask(self, map_data: torch.Tensor) -> torch.Tensor:
        """Ray-casting physical visibility mask.

        Extracts Building Outline (ID=5) and Wall (ID=2) from Ways layer,
        converts to polar, finds first occluder per ray via cumsum.

        Returns: [B, 1, R, Phi], 1.0 = visible, 0.0 = occluded
        """
        device = map_data.device
        B = map_data.shape[0]

        ways_channel = map_data[:, 1]
        occluder_mask = ((ways_channel == 5) | (ways_channel == 2)).float().unsqueeze(1)
        polar_occluder = self.cartesian_to_polar_mask(occluder_mask).squeeze(1)

        occluder_bool = (polar_occluder > 0.5).float()
        cumsum_occ = torch.cumsum(occluder_bool, dim=1)
        first_occ = (cumsum_occ == 1) & (occluder_bool == 1)
        first_occ_dist = torch.argmax(first_occ.float(), dim=1)
        has_occ = torch.any(occluder_bool > 0, dim=1)
        first_occ_dist = torch.where(
            has_occ, first_occ_dist,
            torch.full_like(first_occ_dist, self.radial_resolution - 1))

        radial_idx = torch.arange(self.radial_resolution, device=device)
        radial_idx = radial_idx.view(1, -1, 1).expand(B, -1, self.angular_resolution)
        vis_mask = (radial_idx < first_occ_dist.unsqueeze(1)).float()
        return vis_mask.unsqueeze(1)

    def forward(self, x, return_per_k=False, return_attention=False,
                node_coords_list=None, aug_params_list=None,
                debug_mask=False, saliency_boost_tensor=None, relation_module=None):
        """OSM branch forward.

        Args:
            x: [B, 3, H, W] raw OSM raster
            saliency_boost_tensor: [NUM_NODE_TYPES, C] after softplus, from SFPRModel
            relation_module: optional RelationBiasedLocalAttention for weak residual at x5

        Returns: BEV_model output (descriptor, optionally pooled/attention)
        """
        raw_osm = x

        # 1. Feature Embedding
        embeddings = self.map_encoder(x)

        # 2. Cartesian → Polar
        polar_map = self.cartesian_to_polar(embeddings)

        # 3. Physical visibility mask
        vis_mask = None
        if self.use_physical_visibility:
            vis_mask = self.generate_physical_visibility_mask(raw_osm)
            polar_map = torch.cat([polar_map, vis_mask], dim=1)

        # 4. Saliency modulation at x5
        saliency_modulation_map = None
        if self.use_osm_saliency and node_coords_list is not None:
            if saliency_boost_tensor is None:
                raise ValueError("saliency_boost_tensor required when use_osm_saliency=True")

            B, _, H, W = embeddings.shape
            saliency_field_cart = build_osm_saliency_fields(
                node_coords_list=node_coords_list,
                aug_params_list=aug_params_list,
                H=H, W=W, device=embeddings.device,
                dilation_radius=self.dilation_radius)

            saliency_field_polar = self.cartesian_to_polar_mask(saliency_field_cart)
            x5_h, x5_w = 30, 22
            saliency_field_x5 = F.adaptive_max_pool2d(saliency_field_polar, output_size=(x5_h, x5_w))

            saliency_modulation_map = torch.einsum(
                'bkrp,kc->bcrp', saliency_field_x5,
                saliency_boost_tensor.to(saliency_field_x5.device, dtype=saliency_field_x5.dtype))

            del saliency_field_cart, saliency_field_polar, saliency_field_x5

        # 4.5 Build relation-biased structural attention info for OSM branch (Eq. 8)
        osm_prompting_info = None
        if relation_module is not None and node_coords_list is not None and aug_params_list is not None:
            B_emb, _, H_emb, W_emb = embeddings.shape
            osm_prompting_info = build_osm_prompting_info(
                node_coords_list=node_coords_list,
                aug_params_list=aug_params_list,
                map_H=H_emb, map_W=W_emb,
                tile_size=100.0,
                polar_R=self.radial_resolution,
                polar_Phi=self.angular_resolution,
                x5_H=30, x5_W=22,
                device=embeddings.device,
            )

        # 5. BEV Model
        return self.BEV_model(
            polar_map,
            return_per_k=return_per_k,
            return_attention=return_attention,
            saliency_modulation_map=saliency_modulation_map,
            vis_mask=vis_mask,
            prompting_info=osm_prompting_info,
            relation_module=relation_module)


# =============================================================================
# SFPRModel — Top-Level Cross-Modal Model
# =============================================================================

class SFPRModel(nn.Module):
    """SFPR dual-branch cross-modal place recognition model.

    Implements the structural fingerprint-aware attention mechanism (SFAA, §4.2):
      - Per-type per-channel saliency boost parameters: [NUM_NODE_TYPES, feature_dim]
      - Saliency modulation via einsum(saliency_field, saliency_boost) at x5 resolution
      - Relation-biased structural attention (Eq. 8) for anchor context refinement
      - Fingerprint-aware feature aggregation (FAFA) for global descriptor + attention

    Forward contract:
        Base:       (osm_desc, pc_desc) both [B, D] L2-normalized
        +per_k:     appends (pc_pooled, osm_pooled)
        +attention: appends (pc_attn, osm_attn) — fingerprint-aware attention (Eq. 7)
    """

    def __init__(self, conf,
                 use_osm_saliency: bool = False,
                 osm_node_boost_init: float = 0.5,
                 osm_dilation_radius: float = 3.0,
                 use_pc_saliency: bool = False,
                 pc_node_boost_init: float = 0.1,
                 pc_dilation_r: int = 2,
                 pc_dilation_phi: int = 3,
                 feature_dim: int = 512,
                 use_osm_relation_attention: bool = False,
                 use_pc_relation_attention: bool = False,
                 nlir_hidden_dim: int = 64,
                 nlir_neighbor_radius: float = 5.0,
                 nlir_K_safe: int = 48,
                 nlir_lambda_init: float = 0.02,
                 nlir_lambda_max: float = 0.1):
        super().__init__()

        self.use_osm_saliency = use_osm_saliency
        self.use_pc_saliency = use_pc_saliency
        self.use_osm_relation_attention = use_osm_relation_attention
        self.use_pc_relation_attention = use_pc_relation_attention
        self.pc_dilation_r = pc_dilation_r
        self.pc_dilation_phi = pc_dilation_phi
        self.feature_dim = feature_dim
        self.R, self.Phi = 480, 360
        self.r_min, self.r_max = 3.0, 50.0

        # Per-type boost embeddings (softplus ensures >= 0)
        if use_osm_saliency:
            init_raw = np.log(np.exp(osm_node_boost_init) - 1) if osm_node_boost_init > 0.01 else -2.0
            self.osm_saliency_boost_raw = nn.Parameter(
                torch.full((NUM_NODE_TYPES, feature_dim), init_raw, dtype=torch.float32))

        if use_pc_saliency:
            init_raw = np.log(np.exp(pc_node_boost_init) - 1) if pc_node_boost_init > 0.01 else -3.0
            self.pc_saliency_boost_raw = nn.Parameter(
                torch.full((NUM_NODE_TYPES, feature_dim), init_raw, dtype=torch.float32))

        # ---- Backbone MUST be initialized BEFORE relation attention ----
        # Backbone init before relation attention ensures backbone random weights
        # are identical to baseline (no relation attention). Changes in relation
        # attention param count do not affect backbone RNG state.

        # PC Encoder
        self.pc_encoder = PointCloudBEVEncoder(
            BEV_net=BEVUNetEncoder(
                n_height=33, input_batch_norm=True, circular_padding=True,
                use_wfrm=True, use_x4x5_fusion=False),
            grid_size=[480, 360, 32], fea_dim=24, out_pt_fea_dim=64,
            max_pt_per_encode=256, fea_compre=32)

        # OSM Encoder
        self.osm_encoder = OSMEncoderWrapper(
            conf,
            BEV_net=BEVUNetEncoder(
                n_height=49, input_batch_norm=True, circular_padding=True,
                use_wfrm=False, use_x4x5_fusion=False),
            use_osm_saliency=use_osm_saliency,
            dilation_radius=osm_dilation_radius,
            use_physical_visibility=True)

        # ---- Relation attention modules AFTER backbone ----
        # Initialized after backbone so their random params do not affect backbone RNG state.
        # Toggling or modifying relation attention does not change backbone initialization.
        if use_osm_relation_attention:
            self.osm_relation_attention = RelationBiasedLocalAttention(
                x5_channels=feature_dim, hidden_dim=nlir_hidden_dim,
                num_node_types=NUM_NODE_TYPES,
                neighbor_radius_m=nlir_neighbor_radius,
                K_safe=nlir_K_safe, lambda_init=nlir_lambda_init,
                lambda_max=nlir_lambda_max)

        if use_pc_relation_attention:
            self.pc_relation_attention = RelationBiasedLocalAttention(
                x5_channels=feature_dim, hidden_dim=nlir_hidden_dim,
                num_node_types=NUM_NODE_TYPES,
                neighbor_radius_m=nlir_neighbor_radius,
                K_safe=nlir_K_safe, lambda_init=nlir_lambda_init,
                lambda_max=nlir_lambda_max)

    def get_osm_saliency_boost(self) -> Optional[torch.Tensor]:
        if hasattr(self, 'osm_saliency_boost_raw'):
            return F.softplus(self.osm_saliency_boost_raw)
        return None

    def get_pc_saliency_boost(self) -> Optional[torch.Tensor]:
        if hasattr(self, 'pc_saliency_boost_raw'):
            return F.softplus(self.pc_saliency_boost_raw)
        return None

    def forward(self, osm_data, pc_data, semantic_bev=None,
                return_per_k=False, return_attention=False,
                osm_node_coords_list=None, pc_node_coords_list=None,
                aug_params_list=None, debug_mask=False):
        """Forward pass. See class docstring for return contract.

        Relation-biased structural attention modules are applied internally
        when enabled; they do NOT change
        the external return contract.
        """
        device = osm_data.device

        # ==================== OSM Branch ====================
        osm_saliency_boost = self.get_osm_saliency_boost()
        self._cached_osm_saliency_boost = osm_saliency_boost

        # Relation attention module for OSM (None if not enabled -> no-op in BEVUNetEncoder)
        osm_relation_mod = getattr(self, 'osm_relation_attention', None) if self.use_osm_relation_attention else None

        osm_out = self.osm_encoder(
            osm_data, return_per_k=return_per_k, return_attention=return_attention,
            node_coords_list=osm_node_coords_list,
            aug_params_list=aug_params_list,
            debug_mask=debug_mask, saliency_boost_tensor=osm_saliency_boost,
            relation_module=osm_relation_mod)

        # Unified unpacking for OSM branch
        if isinstance(osm_out, tuple):
            # BEVUNetEncoder returns ordered: [descriptor, ?ring_attn, ?pooled_raw]
            osm_desc = osm_out[0]
            osm_remaining = list(osm_out[1:])
        else:
            osm_desc = osm_out
            osm_remaining = []

        # Extract osm_attn and osm_pooled from remaining
        osm_attn = None
        osm_pooled = None
        idx = 0
        if return_attention and idx < len(osm_remaining):
            osm_attn = osm_remaining[idx]; idx += 1
        if return_per_k and idx < len(osm_remaining):
            osm_pooled = osm_remaining[idx]; idx += 1

        # ==================== PC Branch ====================
        pc_saliency_boost = self.get_pc_saliency_boost()
        self._cached_pc_saliency_boost = pc_saliency_boost

        pc_saliency_modulation_map = None
        if self.use_pc_saliency and pc_node_coords_list is not None and aug_params_list is not None:
            if pc_saliency_boost is None:
                raise ValueError("pc_saliency_boost_raw not initialized but use_pc_saliency=True")

            pc_saliency_fields = build_pc_saliency_fields(
                pc_node_coords_list=pc_node_coords_list,
                pc_aug_params_list=aug_params_list,
                R=self.R, Phi=self.Phi, device=device,
                r_min=self.r_min, r_max=self.r_max,
                dilation_r=self.pc_dilation_r,
                dilation_phi=self.pc_dilation_phi)

            x5_h, x5_w = 30, 22
            pc_saliency_fields_x5 = F.adaptive_max_pool2d(pc_saliency_fields, output_size=(x5_h, x5_w))
            pc_saliency_modulation_map = torch.einsum(
                'bkrp,kc->bcrp', pc_saliency_fields_x5,
                pc_saliency_boost.to(pc_saliency_fields_x5.device, dtype=pc_saliency_fields_x5.dtype))
            del pc_saliency_fields, pc_saliency_fields_x5

        # Build PC relation-biased structural attention info (anchor coords → x5 polar positions)
        pc_relation_mod = getattr(self, 'pc_relation_attention', None) if self.use_pc_relation_attention else None
        pc_prompting_info = None
        if pc_relation_mod is not None and pc_node_coords_list is not None and aug_params_list is not None:
            pc_prompting_info = build_pc_prompting_info(
                pc_node_coords_list=pc_node_coords_list,
                pc_aug_params_list=aug_params_list,
                R=self.R, Phi=self.Phi,
                r_min=self.r_min, r_max=self.r_max,
                x5_H=30, x5_W=22,
                device=device,
            )

        pc_out = self.pc_encoder(
            pc_data[0], pc_data[1], pc_data[2], pc_data[3],
            semantic_bev=semantic_bev, return_per_k=return_per_k,
            return_attention=return_attention, saliency_modulation_map=pc_saliency_modulation_map,
            prompting_info=pc_prompting_info, relation_module=pc_relation_mod)

        # Unified unpacking for PC branch
        if isinstance(pc_out, tuple):
            pc_desc = pc_out[0]
            pc_remaining = list(pc_out[1:])
        else:
            pc_desc = pc_out
            pc_remaining = []

        pc_attn = None
        pc_pooled = None
        idx = 0
        if return_attention and idx < len(pc_remaining):
            pc_attn = pc_remaining[idx]; idx += 1
        if return_per_k and idx < len(pc_remaining):
            pc_pooled = pc_remaining[idx]; idx += 1

        # L2 normalize descriptors
        osm_desc = F.normalize(osm_desc, p=2, dim=1)
        pc_desc = F.normalize(pc_desc, p=2, dim=1)

        # ==================== Build return tuple ====================
        results = [osm_desc, pc_desc]
        if return_per_k:
            results.extend([pc_pooled, osm_pooled])
        if return_attention:
            results.extend([pc_attn, osm_attn])
        return tuple(results)

    def log_saliency_boost_values(self, log_fn=None):
        """Log per-type boost statistics via provided log function."""
        _log = log_fn or logger.info

        if self.use_osm_saliency:
            osm_saliency_boost = self.get_osm_saliency_boost()
            parts = [f"{t}:{osm_saliency_boost[i].mean().item():.3f}/{osm_saliency_boost[i].norm(p=2).item():.1f}"
                     for i, t in enumerate(NODE_SEMANTIC_TYPES)]
            _log(f"  OSM Boost: {' | '.join(parts)}")

        if self.use_pc_saliency:
            pc_saliency_boost = self.get_pc_saliency_boost()
            parts = [f"{t}:{pc_saliency_boost[i].mean().item():.3f}/{pc_saliency_boost[i].norm(p=2).item():.1f}"
                     for i, t in enumerate(NODE_SEMANTIC_TYPES)]
            _log(f"  PC  Boost: {' | '.join(parts)}")

    def log_relation_attention_values(self, log_fn=None):
        """Log relation-biased structural attention statistics.

        Reports λ gate value, δ magnitude, avg anchors and neighbors
        for each enabled branch (OSM / PC).
        """
        _log = log_fn or logger.info

        if self.use_osm_relation_attention and hasattr(self, 'osm_relation_attention'):
            stats = self.osm_relation_attention.log_prompting_stats()
            lmax = self.osm_relation_attention.lambda_max
            _log(f"  OSM RelAttn: λ={stats['lambda_val']:.4f}/{lmax}  "
                 f"δ_abs={stats['delta_abs_mean']:.6f}  "
                 f"avg_nodes={stats['avg_num_nodes']:.1f}  "
                 f"avg_nbrs={stats['avg_num_neighbors']:.1f}")

        if self.use_pc_relation_attention and hasattr(self, 'pc_relation_attention'):
            stats = self.pc_relation_attention.log_prompting_stats()
            lmax = self.pc_relation_attention.lambda_max
            _log(f"  PC  RelAttn: λ={stats['lambda_val']:.4f}/{lmax}  "
                 f"δ_abs={stats['delta_abs_mean']:.6f}  "
                 f"avg_nodes={stats['avg_num_nodes']:.1f}  "
                 f"avg_nbrs={stats['avg_num_neighbors']:.1f}")