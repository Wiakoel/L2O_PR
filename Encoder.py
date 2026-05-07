"""
network/Encoder.py — SFPR Encoder Components

Paper reference: Shared BEV feature extraction (§4.2 Feature Extraction).

Definition order follows dependency (depended-upon first):
  1. Constants / helper functions
  2. UNet building blocks
  3. Semantic-guided feature refinement (WFRM — implementation detail)
  4. X4X5 fusion module (implementation detail)
  5. MapEncoder (OSM raster embedding)
  6. BEVUNetEncoder (uses all of the above)
  7. PointCloudBEVEncoder (uses BEVUNetEncoder)

The encoder supports:
  - Fingerprint-aware saliency modulation at deep feature level (x5)
  - Relation-biased structural attention residual (Eq. 8)
  - Fingerprint-aware feature aggregation for global descriptor (FAFA, Eqs. 5-7)

RESEARCH CORE — do not modify forward math paths.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from network.Spectral_Aggregation import SpectralRingQueryAggregation


# =============================================================================
# 1. Constants
# =============================================================================

WFRM_HL_PRESERVE = {
    9: 0.3, 10: 0.4, 11: 0.2, 12: 0.2, 17: 0.2,       # Ground: suppress
    13: 0.9, 14: 0.7, 18: 0.9, 19: 0.9, 16: 0.9,       # Structure: keep
    15: 0.2,                                              # Vegetation
    0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1,             # Dynamic/Other
    5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1,
}

# Structure class IDs retained by HH (for vectorized ops)
_HH_KEEP_IDS = torch.tensor([13, 14, 16, 18, 19], dtype=torch.long)


# =============================================================================
# 2. Helper Functions
# =============================================================================

def _grp_range_torch(counts: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute group-wise range indices for scatter operations.

    Given counts [c0, c1, ...], returns [0,1,...,c0-1, 0,1,...,c1-1, ...].
    Used to limit max points per voxel.
    """
    idx = torch.cumsum(counts, 0)
    id_arr = torch.ones(idx[-1], dtype=torch.int64, device=device)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -counts[:-1] + 1
    return torch.cumsum(id_arr, 0)


# =============================================================================
# 3. UNet Building Blocks
# =============================================================================

class _DoubleConvCircular(nn.Module):
    """Double 3x3 conv with circular padding on W (angular) dimension."""

    def __init__(self, in_ch, out_ch, group_conv):
        super().__init__()
        g1 = min(out_ch, in_ch) if group_conv else 1
        g2 = out_ch if group_conv else 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0), groups=g1),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0), groups=g2),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(F.pad(x, (1, 1, 0, 0), mode='circular'))
        return self.conv2(F.pad(x, (1, 1, 0, 0), mode='circular'))


class _InConv(nn.Module):
    def __init__(self, in_ch, out_ch, input_batch_norm):
        super().__init__()
        if input_batch_norm:
            self.conv = nn.Sequential(nn.BatchNorm2d(in_ch),
                                      _DoubleConvCircular(in_ch, out_ch, False))
        else:
            self.conv = _DoubleConvCircular(in_ch, out_ch, False)

    def forward(self, x):
        return self.conv(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch, group_conv):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    _DoubleConvCircular(in_ch, out_ch, group_conv))

    def forward(self, x):
        return self.mpconv(x)



# =============================================================================
# 6. MapEncoder (OSM Raster Embedding)
# =============================================================================

class MapEncoder(nn.Module):
    """Embed discrete OSM raster channels into continuous feature maps.

    Each channel (areas, ways, nodes) gets its own nn.Embedding with
    (num_classes + 1) entries to handle unknown/zero labels.

    Output: [B, 3*embedding_dim, H, W]
    """

    def __init__(self, conf):
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {
                k: nn.Embedding(n + 1, conf['embedding_dim'])
                for k, n in conf['num_classes'].items()
            }
        )

    def forward(self, data):
        """
        Args:
            data: [B, 3, H, W] integer tensor (areas, ways, nodes channels)
        Returns:
            embeddings: [B, 3*embedding_dim, H, W]
        """
        embeddings = [
            self.embeddings[k](data[:, i])
            for i, k in enumerate(("areas", "ways", "nodes"))
        ]
        return torch.cat(embeddings, dim=-1).permute(0, 3, 1, 2)


# =============================================================================
# 7. BEVUNetEncoder
# =============================================================================

class BEVUNetEncoder(nn.Module):
    """BEV Encoder with optional WFRM and X4X5 Fusion.

    Forward outputs (ordered):
        global_descriptor: [B, D] (always first)
        x5: [B, C, H5, W5] if return_lf
        ring_attention: [B, R] if return_attention
        pooled_raw: [B, R, C] if return_per_k
    """

    def __init__(self, n_height, dilation=1, group_conv=False, input_batch_norm=True,
                 circular_padding=True, use_wfrm=False, use_x4x5_fusion=False):
        super().__init__()

        self.use_wfrm = use_wfrm
        self.use_x4x5_fusion = use_x4x5_fusion

        self.inc = _InConv(n_height, 32, input_batch_norm)
        self.down1 = _Down(32, 64, group_conv)
        self.down2 = _Down(64, 128, group_conv)
        self.down3 = _Down(128, 256, group_conv)
        self.down4 = _Down(256, 512, group_conv)

        if use_wfrm:
            self.wfrm3 = SemanticGuidedWFRM(channels=128)
            self.wfrm4 = SemanticGuidedWFRM(channels=256)

        if use_x4x5_fusion:
            self.x4x5_fusion = X4X5FusionModule(
                x4_channels=256, x5_channels=512, x4_reduced=128,
                circular_padding=circular_padding)

        self.spectral_aggregation = SpectralRingQueryAggregation(
            in_channels=512, proj_channels=512, num_rqueries=22, num_pqueries=30,
            num_layers=1, row_dim=4, num_freqs=8)

        self.saliency_boost_beta_max = 3.0
        self.saliency_boost_beta_logit = nn.Parameter(torch.tensor(0.6931))

    def forward(self, x, semantic_bev=None, return_lf=False, return_per_k=False,
                return_attention=False, saliency_modulation_map=None, vis_mask=None,
                prompting_info=None, relation_module=None):
        """
        Args:
            x: [B, C, H, W] input features
            saliency_modulation_map: [B, C, H, W] saliency boost map (at x5 resolution)
            vis_mask: [B, 1, R, Phi] physical visibility mask.
                      PC branch must pass None.
            prompting_info: optional dict for relation attention module (default None = no-op)
            relation_module: optional RelationBiasedLocalAttention instance (default None = no-op)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        if self.use_wfrm:
            x3 = self.wfrm3(x3, semantic_bev)

        x4 = self.down3(x3)
        x4_clean = self.wfrm4(x4, semantic_bev) if self.use_wfrm else x4

        x5_raw = self.down4(x4_clean)

        # 1. Fingerprint-aware saliency modulation (AGSP, Eq. 7)
        x5 = x5_raw
        if saliency_modulation_map is not None:
            if saliency_modulation_map.shape[-2:] != x5_raw.shape[-2:]:
                saliency_modulation_map = F.adaptive_max_pool2d(saliency_modulation_map, output_size=x5_raw.shape[-2:])
            if saliency_modulation_map.shape[1] == 1:
                saliency_modulation_map = saliency_modulation_map.expand(-1, x5_raw.shape[1], -1, -1)

            beta = (self.saliency_boost_beta_max * torch.sigmoid(self.saliency_boost_beta_logit)
                    ).to(dtype=saliency_modulation_map.dtype, device=saliency_modulation_map.device).clamp_min(1e-4)
            x5 = x5_raw * (1.0 + beta * torch.tanh(saliency_modulation_map / beta))

        # 2. Relation-biased structural attention: residual from POST-MODULATION x5 (Eq. 8)
        # Token sampling and attention operate on the saliency-modulated feature map,
        # ensuring consistency between boost and relation attention representations.
        # When relation_module or prompting_info is None, degrades to zero residual.
        if relation_module is not None and prompting_info is not None:
            prompting_residual = relation_module(x5, prompting_info)
        else:
            prompting_residual = None

        # 3. Combine: x5 = x5_boosted + prompting_residual
        if prompting_residual is not None:
            x5 = x5 + prompting_residual

        if self.use_x4x5_fusion:
            x5 = self.x4x5_fusion(x4_clean, x5)

        # 4. Visibility Mask for Spectral Decomposition
        vis_mask5 = None
        if vis_mask is not None:
            vis_mask5 = F.adaptive_avg_pool2d(vis_mask.float(), x5.shape[-2:])

        # 5. Spectral Ring Query Aggregation
        if return_per_k:
            global_descriptor, attns, pooled_raw = self.spectral_aggregation(
                x5, mask=vis_mask5, return_per_k=True, return_attention=return_attention)
        else:
            global_descriptor, attns = self.spectral_aggregation(
                x5, mask=vis_mask5, return_attention=return_attention)

        # 6. Fingerprint-aware attention extraction (Eq. 7 — c_i^b)
        ring_attention = None
        if return_attention:
            last_attn = attns[-1]
            ring_attention = last_attn.mean(dim=1)

        # 7. Build return tuple
        results = [global_descriptor]
        if return_lf:
            results.append(x5)
        if return_attention:
            results.append(ring_attention)
        if return_per_k:
            results.append(pooled_raw)

        return results[0] if len(results) == 1 else tuple(results)


# =============================================================================
# 8. PointCloudBEVEncoder
# =============================================================================

class PointCloudBEVEncoder(nn.Module):
    """Point cloud → BEV feature grid → BEV encoder → global descriptor.

    Pipeline: raw points → PPmodel (MLP) → scatter_max → BEV grid → BEVUNetEncoder
    """

    def __init__(self, BEV_net, grid_size, fea_dim=3, out_pt_fea_dim=64,
                 max_pt_per_encode=256, fea_compre=None):
        super().__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            nn.Linear(fea_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_pt_fea_dim),
        )

        self.BEV_model = BEV_net
        self.grid_size = grid_size
        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.semantic_embedding = nn.Embedding(32, 16)
        self.pool_dim = out_pt_fea_dim

        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU(),
            )
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind, pt_label, voxel_fea=None,
                semantic_bev=None, return_per_k=False, return_attention=False,
                saliency_modulation_map=None, prompting_info=None, relation_module=None):
        cur_dev = pt_fea[0].device
        B = len(xy_ind)

        cat_pt_ind = []
        for i in range(B):
            cat_pt_ind.append(F.pad(xy_ind[i], (1, 0), 'constant', value=i))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        cat_pt_label = torch.cat(pt_label, dim=0)
        pt_num = cat_pt_ind.shape[0]

        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]
        cat_pt_label = cat_pt_label[shuffled_ind, :]

        unq, unq_inv, unq_cnt = torch.unique(
            cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)
        grp_ind = _grp_range_torch(unq_cnt, cur_dev)[
            torch.argsort(torch.argsort(unq_inv))]
        remain_ind = grp_ind < self.max_pt

        cat_pt_fea = cat_pt_fea[remain_ind, :]
        cat_pt_ind = cat_pt_ind[remain_ind, :]
        cat_pt_label = cat_pt_label[remain_ind, :]
        unq_inv = unq_inv[remain_ind]

        cat_pt_label = torch.squeeze(self.semantic_embedding(cat_pt_label))
        cat_pt_fea = torch.cat([cat_pt_fea, cat_pt_label], dim=-1)
        processed = self.PPmodel(cat_pt_fea)

        pooled_data = torch_scatter.scatter_max(processed, unq_inv, dim=0)[0]
        del processed, cat_pt_fea, cat_pt_label, unq_inv, grp_ind

        if self.fea_compre:
            pooled_data = self.fea_compression(pooled_data)

        out_data = torch.zeros(
            B, self.grid_size[0], self.grid_size[1], self.pt_fea_dim,
            dtype=torch.float32, device=cur_dev)
        out_data[unq[:, 0], unq[:, 1], unq[:, 2], :] = pooled_data
        out_data = out_data.permute(0, 3, 1, 2)

        del unq, pooled_data

        if voxel_fea is not None:
            out_data = torch.cat((out_data, voxel_fea.unsqueeze(1)), 1)

        return self.BEV_model(
            out_data,
            semantic_bev=semantic_bev,
            return_per_k=return_per_k,
            return_attention=return_attention,
            saliency_modulation_map=saliency_modulation_map,
            vis_mask=None,
            prompting_info=prompting_info,
            relation_module=relation_module,
        )


# =============================================================================
# 4. Components
# =============================================================================

class LearnableSoftThresholding(nn.Module):
    """Learnable soft thresholding for wavelet denoising."""

    def __init__(self, channels, init_threshold=0.1):
        super().__init__()
        self.threshold = nn.Parameter(torch.full((1, channels, 1, 1), init_threshold))

    def forward(self, x):
        t = torch.abs(self.threshold)
        return torch.sign(x) * F.relu(torch.abs(x) - t)


class SemanticGuidedWFRM(nn.Module):
    """Semantic-Guided Wavelet Feature Refinement Module.

    2D Haar wavelet decomposition → semantic-guided coefficient refinement →
    reconstruction. Hybrid padding: circular for W (angular), reflect for H (radial).
    """

    def __init__(self, channels: int, num_classes: int = 20):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.alpha = nn.Parameter(torch.tensor(2.0))

        hl_preserve_base = torch.ones(num_classes)
        for label, preserve in WFRM_HL_PRESERVE.items():
            if label < num_classes:
                hl_preserve_base[label] = preserve
        self.register_buffer('hl_base_weight', hl_preserve_base)
        self.hl_learnable_bias = nn.Parameter(torch.zeros(num_classes))

        self.lh_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
        )
        nn.init.zeros_(self.lh_gate[-1].weight)
        nn.init.constant_(self.lh_gate[-1].bias, 0.5)

        self.hl_refine = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
        )
        nn.init.zeros_(self.hl_refine[-1].weight)
        nn.init.zeros_(self.hl_refine[-1].bias)

        self.hl_threshold = LearnableSoftThresholding(channels, init_threshold=0.1)
        self.hh_threshold = LearnableSoftThresholding(channels, init_threshold=0.2)
        self._init_wavelet_filters()

    def _init_wavelet_filters(self):
        lo = np.array([1, 1]) / np.sqrt(2)
        hi = np.array([-1, 1]) / np.sqrt(2)
        ll = np.outer(lo, lo); lh = np.outer(hi, lo)
        hl = np.outer(lo, hi); hh = np.outer(hi, hi)

        def expand_filter(f, c):
            return torch.tensor(f, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(c, 1, 2, 2).contiguous()

        for name, filt in [('dec_ll', ll), ('dec_lh', lh), ('dec_hl', hl), ('dec_hh', hh),
                           ('rec_ll', ll), ('rec_lh', lh), ('rec_hl', hl), ('rec_hh', hh)]:
            self.register_buffer(name, expand_filter(filt, self.channels))

    def forward(self, x: torch.Tensor, semantic_bev: torch.Tensor = None):
        B, C, H, W = x.shape
        x_input = x
        pad_h, pad_w = H % 2, W % 2

        if pad_w:
            x = F.pad(x, (0, pad_w, 0, 0), mode='circular')
        if pad_h:
            x = F.pad(x, (0, 0, 0, pad_h), mode='reflect')

        ll = F.conv2d(x, self.dec_ll, stride=2, groups=C)
        lh = F.conv2d(x, self.dec_lh, stride=2, groups=C)
        hl = F.conv2d(x, self.dec_hl, stride=2, groups=C)
        hh = F.conv2d(x, self.dec_hh, stride=2, groups=C)

        lh_refined = lh * torch.sigmoid(self.lh_gate(lh))

        hl_denoised = self.hl_threshold(hl)
        hl_pad_w = F.pad(hl, (1, 1, 0, 0), mode='circular')
        hl_padded = F.pad(hl_pad_w, (0, 0, 1, 1), mode='reflect')

        sem_down = None
        if semantic_bev is not None:
            if semantic_bev.dim() == 3:
                semantic_bev = semantic_bev.unsqueeze(1)
            sem_down = F.interpolate(semantic_bev.float(), size=hl.shape[2:],
                                     mode='nearest').long().squeeze(1)

        if sem_down is not None:
            current_weights = (self.hl_base_weight + 0.2 * torch.tanh(self.hl_learnable_bias)).clamp(0.0, 1.0)
            hl_semantic_attn = current_weights[sem_down.clamp(0, self.num_classes - 1)].unsqueeze(1)
            hl_refine_factor = torch.tanh(self.hl_refine(hl_padded)) * 0.1
            hl_refined = hl_denoised * (hl_semantic_attn + hl_refine_factor).clamp(0, 1)
        else:
            hl_refined = hl_denoised * torch.sigmoid(self.hl_refine(hl_padded) + 0.5)

        hh_denoised = self.hh_threshold(hh)
        if sem_down is not None:
            keep_ids = _HH_KEEP_IDS.to(sem_down.device)
            hh_mask_bool = (sem_down.unsqueeze(-1) == keep_ids).any(dim=-1)
            hh_refined = hh_denoised * hh_mask_bool.float().unsqueeze(1) * 0.5
        else:
            hh_refined = hh_denoised * 0.1

        ll_up = F.conv_transpose2d(ll, self.rec_ll, stride=2, groups=C)
        lh_up = F.conv_transpose2d(lh_refined, self.rec_lh, stride=2, groups=C)
        hl_up = F.conv_transpose2d(hl_refined, self.rec_hl, stride=2, groups=C)
        hh_up = F.conv_transpose2d(hh_refined, self.rec_hh, stride=2, groups=C)

        x_wfrm = ll_up + lh_up + hl_up + hh_up
        if pad_h or pad_w:
            x_wfrm = x_wfrm[:, :, :H, :W]

        alpha = torch.sigmoid(self.alpha)
        return x_input + alpha * (x_wfrm - x_input)


# =============================================================================
# 5. Fusion Module
# =============================================================================

class X4X5FusionModule(nn.Module):
    """Lightweight concatenation fusion: x4_down + x5."""

    def __init__(self, x4_channels=256, x5_channels=512, x4_reduced=128,
                 circular_padding=True):
        super().__init__()
        self.x4_adapter = nn.Sequential(
            nn.Conv2d(x4_channels, x4_reduced, kernel_size=3, stride=2,
                      padding=(1, 0), bias=False),
            nn.BatchNorm2d(x4_reduced),
            nn.LeakyReLU(inplace=True),
        )
        fused_channels = x5_channels + x4_reduced
        padding = (1, 0) if circular_padding else 1
        self.fusion_conv = nn.Conv2d(fused_channels, x5_channels,
                                     kernel_size=3, padding=padding, bias=False)
        self.fusion_bn = nn.BatchNorm2d(x5_channels)
        self.fusion_act = nn.LeakyReLU(inplace=True)
        self.circular_padding = circular_padding

    def forward(self, x4, x5):
        x4_down = self.x4_adapter(F.pad(x4, (1, 1, 0, 0), mode='circular'))
        if x4_down.shape[-2:] != x5.shape[-2:]:
            x4_down = x4_down[:, :, :x5.shape[-2], :x5.shape[-1]]

        x_cat = torch.cat([x5, x4_down], dim=1)
        if self.circular_padding:
            x_cat = F.pad(x_cat, (1, 1, 0, 0), mode='circular')

        return self.fusion_act(self.fusion_bn(self.fusion_conv(x_cat)))
