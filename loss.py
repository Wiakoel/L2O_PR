"""
loss.py — Fingerprint-Guided Embedding Space Reshaping (§4.4)

Implements the training-time feedback from structural fingerprint consistency
scores G_ij into the retrieval embedding space.

Sections:
  1. Fingerprint Consistency Circle Loss (FCCL, Eq. 10)
  2. False-Positive Suppression Loss (L_FP, Eq. 11)
  3. Fingerprint Guidance Scheduler
  4. Online Fingerprint Consistency Estimation
  5. Descriptor Bank for Online OSM Candidates
  6. Parallel Fingerprint Matching Pool

Design:
  - Unverified pairs fall back to baseline Circle Loss behavior.
  - Verified pairs are modulated by G_ij:
      High G_ii (positive) → stronger pull toward ground-truth match.
      Low G_ij (negative)  → amplified punishment for false positives.
      High G_ij (negative) → relaxed punishment for structural twins (ρ term).
  - verified_mask separates "not yet verified" from "verified with G=0".
  - Log-sum-exp formulation for numerical stability.
"""

import math
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from network.FingerPrint import SE2FingerprintVerifier, _xy_to_numpy
from utils import worker_init_fn

# =============================================================================
# 1. Fingerprint Consistency Circle Loss (FCCL)
# =============================================================================

class FingerprintConsistencyLoss(nn.Module):
    """Fingerprint-Guided Circle Loss (FCCL, Eq. 10).

    When geo_matrix=None: standard Circle Loss (full backward compatibility).
    When geo_matrix provided: fingerprint consistency-modulated α and Δ.

    consistency_weight controls the modulation strength (ramp-up friendly):
      consistency_weight=0 → exact standard Circle Loss
      consistency_weight=1 → full fingerprint-guided modulation

    Positive modulation (high G_ii → stronger pull, Eq. 9 top):
        α̂_p = [1+m-s]+ · (1 + v_ii · w · β_p · G_ii)
        Δ̂_p = (1 - m) + v_ii · w · η_p · G_ii

    Negative modulation (Eq. 9 bottom):
        fp_factor    = 1 + v_ij · w · β_n · (1 - G_ij)
        relax_factor = 1 - v_ij · w · ρ · G_ij
        α̂_n = [s+m]+ · fp_factor · relax_factor
        Δ̂_n = m + v_ij · w · η_n · (1 - G_ij)

    Where w = consistency_weight, v_ij = verified_mask[i,j].
    When v_ij=0, all factors reduce to 1 → exact baseline Circle Loss.

    Args:
        m:      Circle Loss margin (default 0.25)
        gamma:  Circle Loss scale factor (default 32)
        beta_p: Positive fingerprint consistency reward strength λ_p (default 2.0)
        eta_p:  Positive margin boost η_p (default 0.04)
        beta_n: False-positive amplification strength λ_n (default 2.0)
        rho:    Physical-twin forgiveness strength ρ (default 0.6)
        eta_n:  Negative margin shift strength η_n (default 0.04)
    """

    def __init__(self, m=0.25, gamma=32,
                 beta_p=2.0, eta_p=0.04,
                 beta_n=2.0, rho=0.6, eta_n=0.04):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.beta_p = beta_p
        self.eta_p = eta_p
        self.beta_n = beta_n
        self.rho = rho
        self.eta_n = eta_n

        assert eta_p < m, f"eta_p ({eta_p}) must be < m ({m})"
        assert eta_n < m, f"eta_n ({eta_n}) must be < m ({m})"
        assert 0 <= rho <= 1, f"rho must be in [0,1], got {rho}"

    def forward(self, osm_descriptors, pc_descriptors, geo_matrix=None,
                consistency_weight=1.0, verified_mask=None):
        """
        Args:
            osm_descriptors: [B, D] L2-normalized
            pc_descriptors:  [B, D] L2-normalized
            geo_matrix:      [B, B] fingerprint consistency scores G_ij,
                             or None for standard Circle Loss.
            consistency_weight: float in [0, 1], ramp-up weight.
            verified_mask:   [B, B] binary tensor. 1 = pair was verified.
                             Required when geo_matrix is not None.

        Returns:
            loss: scalar tensor
            diagnostics: dict (only when geo_matrix is not None)
        """
        sim = F.cosine_similarity(
            osm_descriptors.unsqueeze(1),
            pc_descriptors.unsqueeze(0),
            dim=2,
        )  # [B, B]
        B = sim.size(0)
        pos_mask = torch.eye(B, device=sim.device).bool()
        neg_mask = ~pos_mask

        sp = sim[pos_mask].view(B, 1)   # [B, 1]
        sn = sim[neg_mask].view(B, -1)  # [B, B-1]

        # Base Circle Loss α and Δ
        alpha_p = torch.clamp_min(-sp + 1 + self.m, 0.)
        alpha_n = torch.clamp_min(sn + self.m, 0.)
        delta_p = 1 - self.m
        delta_n = self.m

        if geo_matrix is not None:
            assert verified_mask is not None, (
                "verified_mask is required when geo_matrix is provided. "
                "Use compute_batch_fingerprint_consistency() which returns both."
            )
            w = float(consistency_weight)
            gp = geo_matrix[pos_mask].view(B, 1)   # G_ii: [B, 1]
            gn = geo_matrix[neg_mask].view(B, -1)  # G_ij: [B, B-1]

            vp = verified_mask[pos_mask].view(B, 1)
            vn = verified_mask[neg_mask].view(B, -1)

            # Positive modulation: high G_ii → stronger pull
            alpha_p = alpha_p * (1.0 + vp * w * self.beta_p * gp)
            delta_p = (1 - self.m) + vp * w * self.eta_p * gp

            # Negative modulation (Eq. 9 bottom)
            fp_factor = 1.0 + vn * w * self.beta_n * (1.0 - gn)
            relax_factor = 1.0 - vn * w * self.rho * gn
            alpha_n = alpha_n * fp_factor * relax_factor
            delta_n = self.m + vn * w * self.eta_n * (1.0 - gn)

        # Log-sum-exp stable Circle Loss
        logit_p = -self.gamma * alpha_p * (sp - delta_p)
        logit_n = self.gamma * alpha_n * (sn - delta_n)

        log_pos = torch.logsumexp(logit_p, dim=1)
        log_neg = torch.logsumexp(logit_n, dim=1)
        loss = F.softplus(log_pos + log_neg).mean()

        if geo_matrix is not None:
            with torch.no_grad():
                gp_flat = gp.squeeze(-1)
                gn_flat = gn
                vp_flat = vp.squeeze(-1)
                vn_flat = vn

                vn_verified = vn_flat > 0
                vp_verified = vp_flat > 0
                diag = {
                    'fingerprint_pos_mean': gp_flat[vp_verified].mean().item()
                        if vp_verified.any() else 0.0,
                    'fingerprint_pos_verified': vp_verified.sum().item(),
                    'fingerprint_pos_total': B,
                    'fingerprint_neg_verified': vn_verified.sum().item(),
                    'fingerprint_neg_unverified': (~vn_verified).sum().item(),
                    'fingerprint_neg_verified_mean': (
                        gn_flat[vn_verified].mean().item()
                        if vn_verified.any() else 0.0),
                    'fingerprint_neg_high': (gn_flat[vn_verified] > 0.3).sum().item()
                        if vn_verified.any() else 0,
                    'fingerprint_reshaping_weight': w,
                }
            return loss, diag
        return loss


# Backward-compatible alias
CircleLoss = FingerprintConsistencyLoss


# =============================================================================
# 2. False-Positive Suppression Loss (L_FP, Eq. 11)
# =============================================================================

def compute_false_positive_suppression_loss(
    pc_desc,              # [B, D] current batch PC descriptors (GPU, has grad)
    dataset_indices,      # [B] dataset index list
    osm_bank_desc,        # [N, D] frozen OSM descriptor bank
    geo_scores_topk,      # list of B dicts: {cand_idx: (G_ij, is_trusted)} or None
    margin_fp=0.1,
    device=None,
):
    """Top-K False Positive Suppression Loss (Eq. 11).

    For each query i, penalizes candidates j where:
      - j ≠ g(i) (not ground truth)
      - descriptor similarity s_ij is high
      - fingerprint consistency G_ij is low (false positive)

    L_FP = (1/K) Σ_{j∈N_K(i)} max(0, s_ij - s_{i,g(i)} + m_fp) · (1 - G_ij)
    """
    if device is None:
        device = pc_desc.device

    B = pc_desc.size(0)
    ds_indices = [int(idx) for idx in dataset_indices]

    losses = []
    total_fp_count = 0
    total_punished = 0

    for i in range(B):
        geo_info = geo_scores_topk[i] if geo_scores_topk is not None else None
        if geo_info is None or len(geo_info) == 0:
            continue

        gt_idx = ds_indices[i]

        cand_indices = []
        cand_geo = []
        for cand_idx, (geo, trusted) in geo_info.items():
            cand_indices.append(cand_idx)
            cand_geo.append(geo if trusted else 0.0)

        if len(cand_indices) == 0:
            continue

        cand_idx_t = torch.LongTensor(cand_indices)
        cand_geo_t = torch.tensor(cand_geo, dtype=torch.float32, device=device)

        cand_desc = osm_bank_desc[cand_idx_t].to(device).detach()
        sims = cand_desc @ pc_desc[i]

        gt_mask = (cand_idx_t == gt_idx)
        if not gt_mask.any():
            raise RuntimeError(
                f"[FP Suppression] GT index {gt_idx} not found in candidate set "
                f"{cand_indices} for query batch position {i}. "
                f"GT should always be included in the candidate set."
            )
        s_gt = sims[gt_mask].max()

        fp_mask = (cand_idx_t != gt_idx)
        if not fp_mask.any():
            continue

        s_fp = sims[fp_mask]
        g_fp = cand_geo_t[fp_mask]

        # Hinge loss: max(0, s_fp - s_gt + m_fp) · (1 - G_fp)
        hinge = torch.clamp(s_fp - s_gt.detach() + margin_fp, min=0)
        consistency_weight = (1.0 - g_fp)
        fp_loss = (hinge * consistency_weight)

        active = (hinge > 0)
        if active.any():
            losses.append(fp_loss[active].mean())
            total_punished += active.sum().item()

        total_fp_count += fp_mask.sum().item()

    if len(losses) == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {'n_fp_total': 0, 'n_punished': 0, 'n_queries_active': 0}

    loss = torch.stack(losses).mean()
    diag = {
        'n_fp_total': total_fp_count,
        'n_punished': total_punished,
        'n_queries_active': len(losses),
    }
    return loss, diag


# =============================================================================
# 3. Fingerprint Guidance Scheduler
# =============================================================================

class FingerprintConsistencyScheduler:
    """Ramp-up scheduler for fingerprint-guided reshaping weights.

    epoch < start_epoch                   → 0
    start_epoch ≤ epoch < start + ramp    → linear ramp 0 → max_weight
    epoch ≥ start_epoch + ramp            → max_weight
    """

    def __init__(self, start_epoch: int = 5, ramp_epochs: int = 5,
                 max_weight: float = 0.05):
        self.start_epoch = start_epoch
        self.ramp_epochs = max(ramp_epochs, 1)
        self.max_weight = max_weight

    def get_weight(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return 0.0
        progress = (epoch - self.start_epoch + 1) / self.ramp_epochs
        return self.max_weight * min(progress, 1.0)

    def is_active(self, epoch: int) -> bool:
        return epoch >= self.start_epoch

# =============================================================================
# Anchor Coordinate Helpers
# =============================================================================

def _precompute_node_radii(nodes_dict):
    """Compute radii from structural anchor coordinate dict {type: array[N,2]}."""
    out = {}
    for k, v in nodes_dict.items():
        pts = _xy_to_numpy(v)
        out[k] = (np.sqrt(pts[:, 0]**2 + pts[:, 1]**2).astype(np.float32)
                  if pts.shape[0] > 0 else np.zeros(0, dtype=np.float32))
    return out


def _nodes_to_numpy(nodes_dict):
    """Convert anchor dict values to numpy."""
    return {k: _xy_to_numpy(v) for k, v in nodes_dict.items()}


# =============================================================================
# 4. Online Fingerprint Consistency Estimation
# =============================================================================

def compute_batch_fingerprint_consistency(
    batch, pc_attn, osm_attn, sim_matrix, verifier,
    neg_verify_per_query=2, r_max=50.0, num_rings=30,
    pool=None,
):
    """Compute [B, B] fingerprint consistency matrix G and verified_mask.

    For each query-map pair, runs structural fingerprint extraction and matching
    (§4.3) to obtain the fingerprint consistency score G_ij.

    Returns: (fingerprint_consistency_matrix, verified_mask, stats)
    """
    B = sim_matrix.size(0)
    NR = int(num_rings)
    fingerprint_consistency_matrix = torch.zeros(B, B)
    verified_mask = torch.zeros(B, B)

    pc_nodes_list = batch['pc_node_coords_list']
    osm_nodes_list = batch['osm_node_coords_list']

    # Pre-compute ring bucketing for all samples
    pc_bk_cache, pc_attn_cache, pc_order_cache, osm_bk_cache = [], [], [], []

    for i in range(B):
        pc_n = _nodes_to_numpy(pc_nodes_list[i])
        pc_r = _precompute_node_radii(pc_nodes_list[i])
        pa = pc_attn[i] if isinstance(pc_attn[i], np.ndarray) else pc_attn[i].cpu().numpy()
        pc_bk = verifier._bucket(pc_n, pc_r, None, r_max, NR)
        attn_arr = np.asarray(pa, dtype=np.float32).ravel()
        if attn_arr.shape[0] < NR:
            attn_arr = np.pad(attn_arr, (0, NR - attn_arr.shape[0]))
        pc_bk_cache.append(pc_bk)
        pc_attn_cache.append(attn_arr)
        pc_order_cache.append(list(np.argsort(attn_arr)[::-1]))

    for j in range(B):
        osm_n = _nodes_to_numpy(osm_nodes_list[j])
        osm_r = _precompute_node_radii(osm_nodes_list[j])
        osm_bk_cache.append(verifier._bucket(osm_n, osm_r, None, r_max, NR))

    # Collect pairs to verify: all positives + top-K hardest negatives
    pairs = [(i, i) for i in range(B)]
    if neg_verify_per_query > 0:
        sim_np = sim_matrix.detach().cpu()
        for i in range(B):
            neg_sims = sim_np[i].clone()
            neg_sims[i] = -10.0
            k = min(neg_verify_per_query, B - 1)
            _, topk_j = neg_sims.topk(k)
            for j in topk_j.tolist():
                pairs.append((i, j))

    tasks = [(qi, mj, pc_bk_cache[qi], pc_attn_cache[qi],
              pc_order_cache[qi], osm_bk_cache[mj], NR) for qi, mj in pairs]

    results = (pool.map_batch_verify(tasks) if pool is not None
               else [_fingerprint_batch_worker(t) for t in tasks])

    n_verified, n_trusted = 0, 0
    for qi, mj, geo, trusted in results:
        fingerprint_consistency_matrix[qi, mj] = geo if trusted else 0.0
        verified_mask[qi, mj] = 1.0
        n_verified += 1
        if trusted:
            n_trusted += 1

    stats = {
        'n_verified': n_verified,
        'n_trusted': n_trusted,
        'fingerprint_pos_mean': fingerprint_consistency_matrix.diag().mean().item(),
        'n_verified_mask_ones': int(verified_mask.sum().item()),
    }
    return fingerprint_consistency_matrix, verified_mask, stats


# =============================================================================
# Top-K False Positive Fingerprint Verification
# =============================================================================

def compute_topk_false_positive_consistency(
    pc_desc, dataset_indices, bank, verifier,
    batch, pc_attn, topk=10, max_queries=4,
    r_max=50.0, num_rings=30, pool=None,
):
    """Search descriptor bank for Top-K candidates per query, compute G_ij.

    Returns: (geo_scores_topk, osm_bank_desc)
    """
    B = pc_desc.size(0)
    NR = int(num_rings)
    device = pc_desc.device
    ds_indices = [int(idx) for idx in dataset_indices]

    osm_bank_gpu = bank['descs_gpu']
    sims = pc_desc.detach() @ osm_bank_gpu.T

    gt_sims = torch.tensor([sims[i, ds_indices[i]].item() for i in range(B)], device=device)
    top1_vals, top1_ids = sims.max(dim=1)
    hardness = top1_vals - gt_sims
    for i in range(B):
        if top1_ids[i].item() == ds_indices[i]:
            hardness[i] = -1.0

    if not (hardness > 0).any():
        return [None] * B, bank['descs']

    _, query_order = hardness.sort(descending=True)
    selected = [qi for qi in query_order[:min(max_queries, B)].tolist()
                if hardness[qi] > 0]
    if not selected:
        return [None] * B, bank['descs']

    has_osm_bk = ('osm_bk' in bank and bank['osm_bk'] is not None
                  and bank['osm_bk'][0] is not None)

    geo_scores_topk = [None] * B
    tasks, task_query_map = [], []

    for qi in selected:
        gt_idx = ds_indices[qi]
        _, topk_ids = sims[qi].topk(topk)
        cand_set = set(topk_ids.cpu().tolist())
        cand_set.add(gt_idx)
        cand_list = sorted(cand_set)

        pc_n = _nodes_to_numpy(batch['pc_node_coords_list'][qi])
        pc_r = _precompute_node_radii(batch['pc_node_coords_list'][qi])
        pa = pc_attn[qi] if isinstance(pc_attn[qi], np.ndarray) else pc_attn[qi].cpu().numpy()
        pc_bk = verifier._bucket(pc_n, pc_r, None, r_max, NR)

        attn_arr = np.asarray(pa, dtype=np.float32).ravel()
        if attn_arr.shape[0] < NR:
            attn_arr = np.pad(attn_arr, (0, NR - attn_arr.shape[0]))
        ring_order = list(np.argsort(attn_arr)[::-1])

        if has_osm_bk and pool is not None:
            tasks.append((pc_bk, attn_arr, ring_order, cand_list, NR))
            task_query_map.append(qi)
        else:
            geo_info = {}
            for cj in cand_list:
                if has_osm_bk and bank['osm_bk'][cj] is not None:
                    geo, trusted, _ = verifier._run_matching_phases(
                        pc_bk, bank['osm_bk'][cj], attn_arr, ring_order, NR)
                else:
                    osm_n = bank['nodes'][cj]
                    osm_r = bank['radii'][cj]
                    osm_a = bank['attns'][cj]
                    geo, trusted, _ = verifier.match_with_query_cache(
                        pc_bk, pa, osm_n, osm_r, osm_a,
                        r_max=r_max, num_rings=num_rings)
                geo_info[cj] = (geo, trusted)
            geo_scores_topk[qi] = geo_info

    if tasks:
        results = pool.map_fp_verify(tasks)
        for qi, geo_info in zip(task_query_map, results):
            geo_scores_topk[qi] = geo_info

    return geo_scores_topk, bank['descs']

# =============================================================================
# Ring Alignment Loss (auxiliary)
# =============================================================================

def ring_alignment_loss(pc_pooled, osm_pooled, eps=1e-7):
    """Visibility-aware ring-level cross-modal alignment loss.

    Implementation detail — uses ring token energy from both branches
    as confidence proxy. Only rings with signal in both branches contribute.
    """
    pc_energy = pc_pooled.detach().norm(p=2, dim=-1)     # [B, R]
    osm_energy = osm_pooled.detach().norm(p=2, dim=-1)   # [B, R]

    joint_energy = torch.minimum(pc_energy, osm_energy)
    w = joint_energy / joint_energy.max(dim=-1, keepdim=True).values.clamp_min(eps)

    pc_norm = F.normalize(pc_pooled, p=2, dim=-1)
    osm_norm = F.normalize(osm_pooled, p=2, dim=-1)
    cos_sim = (pc_norm * osm_norm).sum(dim=-1)

    loss = ((1.0 - cos_sim) * w).sum() / w.sum().clamp_min(eps)
    return loss


# =============================================================================
# 5. Descriptor Bank for Online OSM Candidates
# =============================================================================

@torch.no_grad()
def build_descriptor_bank(model, train_dataset, device, config, config_for_fwd,
                     collate_fn, prepare_batch_fn, build_forward_kwargs_fn,
                     verifier=None, r_max=50.0, num_rings=30):
    """Build training bank with OSM descriptors + metadata for online fingerprint consistency.

    Used by the fingerprint-guided reshaping module to maintain an OSM
    descriptor bank for computing G_ij during training.
    """
    model.eval()
    original_mode = train_dataset.mode
    train_dataset.mode = 'eval'

    N = len(train_dataset)
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    bank_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False, drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False,
        worker_init_fn=worker_init_fn)

    osm_bank = None
    osm_attns_list = [None] * N
    osm_nodes_list = [None] * N
    osm_radii_list = [None] * N
    osm_bk_list = [None] * N

    for data in bank_loader:
        batch = prepare_batch_fn(data, device)
        kwargs = build_forward_kwargs_fn(batch, config_for_fwd, return_per_k=False,
                                          return_attention=True)

        osm_desc, _, _, osm_attn = model(
            batch['osm_map'], batch['pc_data'], **kwargs)
        osm_desc = F.normalize(osm_desc, p=2, dim=1)

        if osm_bank is None:
            osm_bank = torch.zeros(N, osm_desc.size(1), dtype=torch.float32)

        for j, idx in enumerate(batch['dataset_indices']):
            idx_int = int(idx)
            osm_bank[idx_int] = osm_desc[j].cpu()
            osm_attns_list[idx_int] = osm_attn[j].cpu().numpy().astype(np.float32)

            nodes_np = _nodes_to_numpy(batch['osm_node_coords_list'][j])
            radii_np = _precompute_node_radii(batch['osm_node_coords_list'][j])
            osm_nodes_list[idx_int] = nodes_np
            osm_radii_list[idx_int] = radii_np

            if verifier is not None:
                osm_bk_list[idx_int] = verifier._bucket(
                    nodes_np, radii_np, None, r_max, num_rings)

    train_dataset.mode = original_mode
    model.train()

    return {
        'descs': osm_bank,
        'attns': osm_attns_list,
        'nodes': osm_nodes_list,
        'radii': osm_radii_list,
        'osm_bk': osm_bk_list,
        'update_count': np.zeros(N, dtype=np.int32),
    }


@torch.no_grad()
def update_descriptor_bank_ema(model, bank, batch, batch_fwd_out, config,
                          device, verifier=None, ema_mu=0.3,
                          r_max=50.0, num_rings=30):
    """EMA bank update: b_i ← μ·b_i + (1−μ)·d_i for current batch.

    Returns: number of entries updated.
    """
    mu = float(ema_mu)

    osm_desc_batch = F.normalize(batch_fwd_out[0].detach(), p=2, dim=1)
    osm_attn_batch = batch_fwd_out[-1]

    indices = batch['dataset_indices']
    batch_indices = set()

    for j, idx in enumerate(indices):
        idx_int = int(idx)
        batch_indices.add(idx_int)

        old_desc = bank['descs'][idx_int]
        new_desc = osm_desc_batch[j].cpu()
        bank['descs'][idx_int] = mu * old_desc + (1 - mu) * new_desc

        bank['attns'][idx_int] = osm_attn_batch[j].detach().cpu().numpy().astype(np.float32)

        if bank['nodes'][idx_int] is None:
            nodes_np = _nodes_to_numpy(batch['osm_node_coords_list'][j])
            radii_np = _precompute_node_radii(batch['osm_node_coords_list'][j])
            bank['nodes'][idx_int] = nodes_np
            bank['radii'][idx_int] = radii_np
            if verifier is not None:
                bank['osm_bk'][idx_int] = verifier._bucket(
                    nodes_np, radii_np, None, r_max, num_rings)

        bank['update_count'][idx_int] += 1

    if 'descs_gpu' in bank:
        for idx_int in batch_indices:
            bank['descs_gpu'][idx_int] = bank['descs'][idx_int].to(device)

    return len(batch_indices)


# =============================================================================
# 6. Parallel Fingerprint Matching Pool
# =============================================================================

_FINGERPRINT_VERIFIER = None
_FINGERPRINT_BANK_OSM_BK = None


def _fingerprint_batch_worker(task):
    """Worker for in-batch fingerprint consistency matrix computation."""
    qi, mj, pc_bk, attn_arr, ring_order, osm_bk, NR = task
    geo, trusted, _ = _FINGERPRINT_VERIFIER._run_matching_phases(
        pc_bk, osm_bk, attn_arr, ring_order, NR)
    return qi, mj, float(geo), bool(trusted)


def _fingerprint_fp_query_worker(task):
    """Worker for false-positive top-K fingerprint verification."""
    pc_bk, attn_arr, ring_order, cand_list, NR = task
    results = {}
    for cj in cand_list:
        osm_bk = _FINGERPRINT_BANK_OSM_BK[cj]
        if osm_bk is None:
            results[cj] = (0.0, False)
            continue
        geo, trusted, _ = _FINGERPRINT_VERIFIER._run_matching_phases(
            pc_bk, osm_bk, attn_arr, ring_order, NR)
        results[cj] = (float(geo), bool(trusted))
    return results


class FingerprintMatchingPool:
    """Persistent multiprocessing pool for structural fingerprint matching.

    Uses fork context: globals shared via copy-on-write.
    """

    def __init__(self, verifier, bank_osm_bk, n_workers=8):
        global _FINGERPRINT_VERIFIER, _FINGERPRINT_BANK_OSM_BK
        _FINGERPRINT_VERIFIER = verifier
        _FINGERPRINT_BANK_OSM_BK = bank_osm_bk
        self.n_workers = max(1, int(n_workers))
        self._active = False

        if self.n_workers > 1:
            ctx = mp.get_context('fork')
            self.pool = ctx.Pool(self.n_workers)
            self._active = True

    def map_batch_verify(self, tasks):
        if not self._active or len(tasks) == 0:
            return [_fingerprint_batch_worker(t) for t in tasks]
        cs = max(1, len(tasks) // self.n_workers)
        return self.pool.map(_fingerprint_batch_worker, tasks, chunksize=cs)

    def map_fp_verify(self, tasks):
        if not self._active or len(tasks) == 0:
            return [_fingerprint_fp_query_worker(t) for t in tasks]
        return self.pool.map(_fingerprint_fp_query_worker, tasks, chunksize=1)

    def close(self):
        if self._active:
            self.pool.close()
            self.pool.join()
            self._active = False

    def __del__(self):
        self.close()


# Backward-compatible alias
FingerprintVerifyPool = FingerprintMatchingPool