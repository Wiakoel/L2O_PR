"""
train_kitti.py — SFPR Training Script

Sections:
  1. Imports
  2. Configuration Migration
  3. Batch Preparation
  4. Forward Argument Construction
  5. Retrieval Validation
  6. Training Epoch
  7. Fingerprint-Guided Reshaping Context Setup
  8. Checkpointing & Main Orchestrator
  9. Setup Helpers
  10. Entry Point
"""

import os
import gc
import time
import argparse
import logging
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR
from tqdm import tqdm

from loss import (
    FingerprintConsistencyLoss,
    FingerprintConsistencyScheduler, compute_false_positive_suppression_loss,
    build_descriptor_bank,
    ring_alignment_loss, update_descriptor_bank_ema,
    compute_batch_fingerprint_consistency, compute_topk_false_positive_consistency,
    FingerprintMatchingPool,
)
from train_kitti_loader import PcMapLocDataset, collate_fn_BEV
from network.Model import SFPRModel, NODE_SEMANTIC_TYPES
from network.FingerPrint import SE2FingerprintVerifier
from utils import (
    set_seed, RNGStateGuard, worker_init_fn, ExperimentLogger,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Migration
# =============================================================================

def normalize_config_keys(config):
    """Migrate legacy config key names to paper-aligned names.

    Supports old configs by mapping:
      saliency_prompting           -> structural_fingerprint_aware_attention
      fingerprint_consistency      -> fingerprint_guided_reshaping
      fcl_gamma / fcl_margin       -> fccl_gamma / fccl_margin
      fcl_max_weight               -> fccl_max_weight
      fp_max_weight                -> fp_suppression_max_weight

    Returns the config dict (modified in place).
    """
    # Top-level section renames
    if 'saliency_prompting' in config and 'structural_fingerprint_aware_attention' not in config:
        config['structural_fingerprint_aware_attention'] = config.pop('saliency_prompting')

    if 'fingerprint_consistency' in config and 'fingerprint_guided_reshaping' not in config:
        config['fingerprint_guided_reshaping'] = config.pop('fingerprint_consistency')

    # Loss key renames
    loss = config.get('loss', {})
    if 'fcl_gamma' in loss and 'fccl_gamma' not in loss:
        loss['fccl_gamma'] = loss.pop('fcl_gamma')
    if 'fcl_margin' in loss and 'fccl_margin' not in loss:
        loss['fccl_margin'] = loss.pop('fcl_margin')

    # Fingerprint-guided reshaping key renames
    fg = config.get('fingerprint_guided_reshaping', {})
    if 'fcl_max_weight' in fg and 'fccl_max_weight' not in fg:
        fg['fccl_max_weight'] = fg.pop('fcl_max_weight')
    if 'fp_max_weight' in fg and 'fp_suppression_max_weight' not in fg:
        fg['fp_suppression_max_weight'] = fg.pop('fp_max_weight')

    return config


# =============================================================================
# Batch Preparation
# =============================================================================

def prepare_batch(data, device):
    """Unpack collated batch and move tensors to device."""
    (grid_ind, point_label, xyz, osm_map, xy, pc_vis_mask,
     intensity, semantic_bev, osm_node_coords_list,
     pc_node_coords_list, aug_params_list, dataset_indices) = data

    pt_fea = [torch.from_numpy(i).float().to(device) for i in xyz]
    grid_ten = [torch.from_numpy(i[:, :2]).to(device) for i in grid_ind]
    pt_label = [torch.from_numpy(i).to(device) for i in point_label]

    return {
        'pc_data': (pt_fea, grid_ten, pt_label, pc_vis_mask.to(device)),
        'osm_map': osm_map.to(device),
        'xy': xy.to(device),
        'semantic_bev': semantic_bev.to(device),
        'osm_node_coords_list': osm_node_coords_list,
        'pc_node_coords_list': pc_node_coords_list,
        'aug_params_list': aug_params_list,
        'dataset_indices': dataset_indices,
    }


def build_forward_kwargs(batch, config, return_per_k=False, return_attention=False,
                         debug=False):
    """Build kwargs dict for model.forward()."""
    sfaa = config['structural_fingerprint_aware_attention']
    return {
        'semantic_bev': batch['semantic_bev'],
        'return_per_k': return_per_k,
        'return_attention': return_attention,
        'aug_params_list': batch['aug_params_list'],
        'debug_mask': debug,
        'osm_node_coords_list': (
            batch['osm_node_coords_list'] if sfaa['osm']['enabled'] else None),
        'pc_node_coords_list': (
            batch['pc_node_coords_list'] if sfaa['pc']['enabled'] else None),
    }


# =============================================================================
# Retrieval Validation
# =============================================================================

def compute_retrieval_metrics(pc_features, osm_database, xy_list):
    """Compute Top-1/5 recall and geo-localization metrics."""
    n = len(pc_features)
    pc_norm = F.normalize(pc_features, p=2, dim=1)
    osm_norm = F.normalize(osm_database, p=2, dim=1)
    sim = pc_norm @ osm_norm.T

    _, top5 = torch.topk(sim, k=min(5, n), dim=1)
    top1 = top5[:, 0]
    correct = torch.arange(n, device=sim.device)

    xy_np = xy_list.cpu().numpy()
    dist = np.linalg.norm(xy_np - xy_np[top1.cpu().numpy()], axis=1)

    return {
        'top_1_ratio': (top1 == correct).sum().item() / n,
        'top_5_ratio': (top5 == correct.unsqueeze(1)).any(dim=1).sum().item() / n,
        'geo_1m': (dist <= 1).sum() / n,
        'geo_5m': (dist <= 5).sum() / n,
        'geo_10m': (dist <= 10).sum() / n,
    }


def validate_retrieval(model, val_loader, device, config):
    """Run retrieval-only validation with RNG isolation.

    Computes descriptor retrieval and localization metrics only.
    No validation loss is computed — the training criterion is not
    meaningful as a validation objective (see §4.4 of the paper).
    """
    with RNGStateGuard(device):
        model.eval()
        osm_db, pc_feats, xys = [], [], []

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation", ncols=100, leave=False):
                batch = prepare_batch(data, device)
                kwargs = build_forward_kwargs(batch, config, return_per_k=False)
                osm_desc, pc_desc = model(batch['osm_map'], batch['pc_data'], **kwargs)
                osm_db.append(osm_desc)
                pc_feats.append(pc_desc)
                xys.append(batch['xy'])

        osm_db = torch.cat(osm_db, dim=0)
        pc_feats = torch.cat(pc_feats, dim=0)
        xys = torch.cat(xys, dim=0)
        retrieval = compute_retrieval_metrics(pc_feats, osm_db, xys)

    return retrieval


def save_checkpoint(state, filename, log=None):
    torch.save(state, filename)
    if log:
        log.info(f"  -> Saved: {os.path.basename(filename)}")


# =============================================================================
# Training Epoch
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    config, epoch, log, fingerprint_ctx=None):
    model.train()
    loss_cfg = config['loss']
    fg_cfg = config['fingerprint_guided_reshaping']
    sfaa_cfg = config['structural_fingerprint_aware_attention']
    ring_align_weight = loss_cfg.get('ring_align_weight', 0.0)

    running = {'total': 0.0, 'fccl': 0.0, 'fp_sup': 0.0, 'ra': 0.0}
    consistency_stats = {
        'n_verified': 0, 'n_trusted': 0, 'n_fp_punished': 0,
        'n_batches': 0, 'fingerprint_pos_mean_sum': 0.0, 'n_bank_updated': 0,
    }

    samples = 0
    lr = optimizer.param_groups[0]['lr']
    fingerprint_active = (fingerprint_ctx is not None and fingerprint_ctx['fccl_weight'] > 0)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [lr={lr:.1e}]", ncols=120)

    for step, data in enumerate(pbar):
        batch = prepare_batch(data, device)
        bs = batch['osm_map'].size(0)
        optimizer.zero_grad()

        need_per_k = (ring_align_weight > 0)
        need_attn = fingerprint_active
        kwargs = build_forward_kwargs(
            batch, config, return_per_k=need_per_k,
            return_attention=need_attn, debug=False)

        fwd_out = model(batch['osm_map'], batch['pc_data'], **kwargs)

        osm_desc, pc_desc = fwd_out[0], fwd_out[1]
        idx = 2
        pc_pooled, osm_pooled = None, None
        if need_per_k:
            pc_pooled = fwd_out[idx]; idx += 1
            osm_pooled = fwd_out[idx]; idx += 1
        pc_attn_t, osm_attn_t = None, None
        if need_attn:
            pc_attn_t = fwd_out[idx]; idx += 1
            osm_attn_t = fwd_out[idx]; idx += 1

        # ── Fingerprint Consistency Circle Loss (FCCL, Eq. 10) ──
        fingerprint_consistency_matrix, verified_mask, batch_consistency_stats = None, None, None
        if fingerprint_active:
            with torch.no_grad():
                sim_sel = F.cosine_similarity(
                    osm_desc.unsqueeze(1), pc_desc.unsqueeze(0), dim=2)
                pc_attn_np = pc_attn_t.detach().cpu().numpy()
                osm_attn_np = osm_attn_t.detach().cpu().numpy()

            fingerprint_consistency_matrix_cpu, verified_mask_cpu, batch_consistency_stats = compute_batch_fingerprint_consistency(
                batch, pc_attn_np, osm_attn_np, sim_sel,
                fingerprint_ctx['verifier'],
                neg_verify_per_query=fg_cfg['neg_verify_per_query'],
                pool=fingerprint_ctx.get('pool'))
            fingerprint_consistency_matrix = fingerprint_consistency_matrix_cpu.to(device)
            verified_mask = verified_mask_cpu.to(device)

        if fingerprint_consistency_matrix is not None:
            fccl_loss, _ = criterion(
                osm_desc, pc_desc, fingerprint_consistency_matrix,
                consistency_weight=fingerprint_ctx['fccl_weight'],
                verified_mask=verified_mask)
        else:
            fccl_loss = criterion(osm_desc, pc_desc)

        # ── False-Positive Suppression (L_FP, Eq. 11) ──
        fp_suppression_loss = torch.tensor(0.0, device=device)
        fp_diag = None
        if fingerprint_active and fingerprint_ctx['fp_suppression_weight'] > 0:
            geo_scores_topk, osm_bank_desc = compute_topk_false_positive_consistency(
                pc_desc, batch['dataset_indices'],
                fingerprint_ctx['bank'], fingerprint_ctx['verifier'],
                batch, pc_attn_np,
                topk=fg_cfg['false_positive_topk'],
                max_queries=fg_cfg['max_queries'],
                pool=fingerprint_ctx.get('pool'))

            fp_suppression_loss, fp_diag = compute_false_positive_suppression_loss(
                pc_desc, batch['dataset_indices'],
                osm_bank_desc, geo_scores_topk,
                margin_fp=fg_cfg['false_positive_margin'], device=device)

        # ── Total Loss (Eq. 12) ──
        ra_loss = torch.tensor(0.0, device=device)
        if need_per_k and pc_pooled is not None and osm_pooled is not None:
            ra_loss = ring_alignment_loss(pc_pooled, osm_pooled)

        fp_sup_weighted = fingerprint_ctx['fp_suppression_weight'] * fp_suppression_loss if fingerprint_active else 0.0
        total_loss = fccl_loss + fp_sup_weighted + ring_align_weight * ra_loss

        if batch_consistency_stats is not None:
            consistency_stats['n_verified'] += batch_consistency_stats['n_verified']
            consistency_stats['n_trusted'] += batch_consistency_stats['n_trusted']
            consistency_stats['fingerprint_pos_mean_sum'] += batch_consistency_stats['fingerprint_pos_mean']
            consistency_stats['n_batches'] += 1
        if fp_diag is not None:
            consistency_stats['n_fp_punished'] += fp_diag.get('n_punished', 0)

        if fingerprint_active and step == 0 and batch_consistency_stats is not None:
            log.info(f"  [FCCL] First batch: verified={batch_consistency_stats['n_verified']}, "
                     f"trusted={batch_consistency_stats['n_trusted']}, "
                     f"fingerprint_pos_mean={batch_consistency_stats['fingerprint_pos_mean']:.4f}")

        total_loss.backward()

        # Gradient check (epoch 0, step 0 only — verifies saliency parameters receive gradients)
        if step == 0 and epoch == 0:
            if sfaa_cfg['osm']['enabled']:
                assert model.osm_saliency_boost_raw.grad is not None, "osm_boost_raw has no gradient!"
                log.info(f"[Grad Check] osm_boost_raw.grad norm: "
                         f"{model.osm_saliency_boost_raw.grad.norm():.6f}")
            if sfaa_cfg['pc']['enabled']:
                assert model.pc_saliency_boost_raw.grad is not None, "pc_boost_raw has no gradient!"
                log.info(f"[Grad Check] pc_boost_raw.grad norm: "
                         f"{model.pc_saliency_boost_raw.grad.norm():.6f}")
            ra_cfg = sfaa_cfg.get('relation_attention', {})
            if ra_cfg.get('osm_enabled', False) and hasattr(model, 'osm_relation_attention'):
                lam = model.osm_relation_attention.attention_gate_raw
                grad_str = f"{lam.grad.norm():.6f}" if lam.grad is not None else "None"
                log.info(f"[Grad Check] osm_relation_attention_gate_raw.grad norm: {grad_str}")
            if ra_cfg.get('pc_enabled', False) and hasattr(model, 'pc_relation_attention'):
                lam = model.pc_relation_attention.attention_gate_raw
                grad_str = f"{lam.grad.norm():.6f}" if lam.grad is not None else "None"
                log.info(f"[Grad Check] pc_relation_attention_gate_raw.grad norm: {grad_str}")

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config['training'].get('max_grad_norm', 1.0))
        optimizer.step()

        # EMA descriptor bank update
        if fingerprint_active and fingerprint_ctx['bank'] is not None and need_attn:
            n_upd = update_descriptor_bank_ema(
                model, fingerprint_ctx['bank'], batch, fwd_out, config,
                device, verifier=fingerprint_ctx['verifier'], ema_mu=0.5)
            consistency_stats['n_bank_updated'] += n_upd

        samples += bs
        running['total'] += total_loss.item() * bs
        running['fccl'] += fccl_loss.item() * bs
        running['ra'] += ra_loss.item() * bs
        running['fp_sup'] += fp_suppression_loss.item() * bs
        pbar.set_postfix({'Loss': f"{running['total'] / samples:.4f}"})

    N = len(train_loader.dataset)
    result = {k: v / N for k, v in running.items()}
    if consistency_stats['n_batches'] > 0:
        nb = consistency_stats['n_batches']
        result.update({
            'fingerprint_n_verified': consistency_stats['n_verified'],
            'fingerprint_n_trusted': consistency_stats['n_trusted'],
            'fingerprint_n_fp_punished': consistency_stats['n_fp_punished'],
            'fingerprint_pos_mean': consistency_stats['fingerprint_pos_mean_sum'] / nb,
            'fingerprint_n_bank_updated': consistency_stats['n_bank_updated'],
        })
    return result


# =============================================================================
# Main Training Orchestrator
# =============================================================================

def train_model(model, train_loader, val_loader, criterion,
                optimizer, scheduler, config, train_dataset=None):
    """Main training orchestrator."""
    device = config['training']['device']
    num_epochs = config['training']['num_epochs']
    loss_cfg = config['loss']
    sfaa_cfg = config['structural_fingerprint_aware_attention']
    fg_cfg = config['fingerprint_guided_reshaping']
    use_fingerprint_guidance = fg_cfg['enabled']
    ring_align_weight = loss_cfg['ring_align_weight']

    # Experiment directory
    exp_name = "sfpr"
    if sfaa_cfg['osm']['enabled']:
        exp_name += f"_osm{sfaa_cfg['osm']['boost_init']}"
    if sfaa_cfg['pc']['enabled']:
        exp_name += f"_pc{sfaa_cfg['pc']['boost_init']}"
    if use_fingerprint_guidance:
        exp_name += "_fg"
    exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    checkpoint_dir = os.path.join(config['training']['checkpoint_dir'], exp_name)
    log = ExperimentLogger(checkpoint_dir)
    log.write_model_config(config, model,
                           len(train_loader.dataset), len(val_loader.dataset))

    best_geo_1m = 0.0
    best_geo_5m, best_geo_10m = 0.0, 0.0
    best_geo1m_epoch = -1
    best_geo5m_epoch, best_geo10m_epoch = -1, -1
    prev_best_geo1m_file = None
    prev_best_geo5m_file, prev_best_geo10m_file = None, None

    log.section(f"Experiment: {exp_name}")
    log.kv("Checkpoint", checkpoint_dir)
    log.kv("Device", device)
    log.kv("Epochs", num_epochs)
    log.kv("Dataset", f"Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}")

    # Fingerprint-Guided Reshaping Setup (§4.4)
    consistency_scheduler, fp_suppression_scheduler, fingerprint_verifier = None, None, None
    if use_fingerprint_guidance:
        if train_dataset is None:
            raise ValueError("train_dataset required when fingerprint_guided_reshaping.enabled=true")
        log.section("Fingerprint-Guided Reshaping")
        consistency_scheduler = FingerprintConsistencyScheduler(
            start_epoch=fg_cfg['start_epoch'],
            ramp_epochs=fg_cfg['ramp_epochs'],
            max_weight=fg_cfg['fccl_max_weight'])
        fp_suppression_scheduler = FingerprintConsistencyScheduler(
            start_epoch=fg_cfg['start_epoch'],
            ramp_epochs=fg_cfg['ramp_epochs'],
            max_weight=fg_cfg['fp_suppression_max_weight'])

        v_cfg = fg_cfg['verifier']
        fingerprint_verifier = SE2FingerprintVerifier(
            verify_thresh=v_cfg['consistency_thresh'],
            graph_thresh=v_cfg['graph_thresh'],
            lambda_attn=v_cfg['saliency_weight'],
            beam_width=v_cfg['beam_width'])

        log.kv("Start Epoch", fg_cfg['start_epoch'])
        log.kv("Ramp Epochs", fg_cfg['ramp_epochs'])
        log.kv("FCCL Max Weight", fg_cfg['fccl_max_weight'])
        log.kv("FP Suppression Max Weight", fg_cfg['fp_suppression_max_weight'])
        log.kv("Fingerprint Workers", fg_cfg['workers'])
    else:
        log.kv("Fingerprint-Guided Reshaping", "OFF")

    log.info("")
    if sfaa_cfg['osm']['enabled'] or sfaa_cfg['pc']['enabled']:
        log.info("[Init] Saliency Boost Values:")
        model.log_saliency_boost_values(log.info)
        log.info("")

    ra_cfg = sfaa_cfg.get('relation_attention', {})
    if ra_cfg.get('osm_enabled', False) or ra_cfg.get('pc_enabled', False):
        log.info("[Init] Relation-Biased Structural Attention:")
        model.log_relation_attention_values(log.info)
        log.info(f"  Config: hidden_dim={ra_cfg.get('hidden_dim', 64)}, "
                 f"neighbor_radius={ra_cfg.get('neighbor_radius', 5.0)}m, "
                 f"K_safe={ra_cfg.get('K_safe', 48)}, "
                 f"lambda_init={ra_cfg.get('lambda_init', 0.005)}, "
                 f"lambda_max={ra_cfg.get('lambda_max', 0.03)}")
        log.info("")

    # ======================== Epoch Loop ========================
    descriptor_bank = None

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        fingerprint_ctx = None

        if use_fingerprint_guidance:
            fccl_w = consistency_scheduler.get_weight(epoch)
            fp_sup_w = fp_suppression_scheduler.get_weight(epoch)

            if fccl_w > 0:
                if descriptor_bank is None:
                    t_bank = time.time()
                    gc.collect()
                    with RNGStateGuard(device):
                        descriptor_bank = build_descriptor_bank(
                            model, train_dataset, device, config, config,
                            collate_fn=collate_fn_BEV,
                            prepare_batch_fn=prepare_batch,
                            build_forward_kwargs_fn=build_forward_kwargs,
                            verifier=fingerprint_verifier)
                    descriptor_bank['descs_gpu'] = descriptor_bank['descs'].to(device)
                    log.info(f"\n[Epoch {epoch:02d}] Fingerprint reshaping: fccl_w={fccl_w:.4f}, "
                             f"fp_sup_w={fp_sup_w:.4f}, bank FULL BUILD "
                             f"[{descriptor_bank['descs'].shape}] in {time.time() - t_bank:.1f}s")
                else:
                    descriptor_bank['descs_gpu'] = descriptor_bank['descs'].to(device)
                    stale = int((descriptor_bank['update_count'] == 0).sum())
                    log.info(f"\n[Epoch {epoch:02d}] Fingerprint reshaping: fccl_w={fccl_w:.4f}, "
                             f"fp_sup_w={fp_sup_w:.4f}, bank EMA (stale={stale}/"
                             f"{len(descriptor_bank['update_count'])})")
            else:
                log.info(f"\n[Epoch {epoch:02d}] Fingerprint reshaping: OFF (before start_epoch)")

            fingerprint_ctx = {
                'bank': descriptor_bank, 'verifier': fingerprint_verifier,
                'scheduler': consistency_scheduler,
                'fccl_weight': fccl_w, 'fp_suppression_weight': fp_sup_w, 'pool': None,
            }
            if fccl_w > 0 and descriptor_bank is not None and fg_cfg['workers'] > 1:
                fingerprint_ctx['pool'] = FingerprintMatchingPool(
                    fingerprint_verifier, descriptor_bank['osm_bk'], n_workers=fg_cfg['workers'])

        # Train
        losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            config, epoch, log, fingerprint_ctx=fingerprint_ctx)

        # Close pool
        if fingerprint_ctx is not None and fingerprint_ctx.get('pool') is not None:
            fingerprint_ctx['pool'].close()

        # Log training losses
        log_msg = (f"\n[Epoch {epoch:02d}] Train  loss={losses['total']:.4f}  "
                f"(fccl={losses['fccl']:.4f}, "
                f"ra={losses['ra']:.6f}, "
                f"fp_sup={losses['fp_sup']:.6f}, lr={current_lr:.2e})")

        if use_fingerprint_guidance and 'fingerprint_n_verified' in losses:
            log_msg += (f"\n           fingerprint: verified={losses['fingerprint_n_verified']}, "
                        f"trusted={losses['fingerprint_n_trusted']}, "
                        f"fp_punished={losses['fingerprint_n_fp_punished']}, "
                        f"pos_mean={losses.get('fingerprint_pos_mean', 0):.4f}, "
                        f"bank_ema_updated={losses.get('fingerprint_n_bank_updated', 0)}")
        log.info(log_msg)

        model.log_saliency_boost_values(log.info)
        model.log_relation_attention_values(log.info)

        # Validate (retrieval metrics only — no val_loss)
        val = validate_retrieval(model, val_loader, device, config)
        log.info(f"[Epoch {epoch:02d}] Val    "
                 f"Geo: 1m={val['geo_1m'] * 100:.2f}%  "
                 f"5m={val['geo_5m'] * 100:.2f}%  "
                 f"10m={val['geo_10m'] * 100:.2f}%  "
                 f"Top1={val['top_1_ratio'] * 100:.1f}%")

        if scheduler is not None:
            scheduler.step()

        # Checkpoint (retrieval metrics only)
        ckpt = {
            'epoch': epoch, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'geo_1m': val['geo_1m'], 'geo_5m': val['geo_5m'],
            'geo_10m': val['geo_10m'],
            'top_1_ratio': val['top_1_ratio'],
            'top_5_ratio': val['top_5_ratio'],
            'config': config,
        }

        if val['geo_1m'] > best_geo_1m:
            best_geo_1m = val['geo_1m']
            best_geo1m_epoch = epoch
            fname = f"best_geo1m_{epoch}.pth"
            fpath = os.path.join(log.model_dir, fname)
            log.info(f"  New Best Geo@1m: {best_geo_1m * 100:.2f}%")
            if prev_best_geo1m_file and os.path.exists(prev_best_geo1m_file):
                os.remove(prev_best_geo1m_file)
            save_checkpoint(ckpt, fpath, log)
            prev_best_geo1m_file = fpath
            log.write_best_update(
                'Geo@1m', epoch, best_geo_1m, fname,
                f"5m={val['geo_5m'] * 100:.1f}%, 10m={val['geo_10m'] * 100:.1f}%")

        if val['geo_5m'] > best_geo_5m:
            best_geo_5m = val['geo_5m']
            best_geo5m_epoch = epoch

        if val['geo_10m'] > best_geo_10m:
            best_geo_10m = val['geo_10m']
            best_geo10m_epoch = epoch

    log.section("Training Finished!")
    log.kv("Best Geo@1m", f"{best_geo_1m * 100:.2f}% (epoch {best_geo1m_epoch})")
    log.kv("Best Geo@5m", f"{best_geo_5m * 100:.2f}% (epoch {best_geo5m_epoch})")
    log.kv("Best Geo@10m", f"{best_geo_10m * 100:.2f}% (epoch {best_geo10m_epoch})")
    log.kv("Checkpoints", checkpoint_dir)
    log.close()


# =============================================================================
# Setup Helpers
# =============================================================================

def build_model(config, log=None):
    _log = log.info if log else logger.info
    sfaa = config['structural_fingerprint_aware_attention']
    ra_cfg = sfaa.get('relation_attention', {})
    model = SFPRModel(
        config,
        use_osm_saliency=sfaa['osm']['enabled'],
        osm_node_boost_init=sfaa['osm']['boost_init'],
        osm_dilation_radius=sfaa['osm']['dilation_radius'],
        use_pc_saliency=sfaa['pc']['enabled'],
        pc_node_boost_init=sfaa['pc']['boost_init'],
        pc_dilation_r=sfaa['pc']['dilation_r'],
        pc_dilation_phi=sfaa['pc']['dilation_phi'],
        use_osm_relation_attention=ra_cfg.get('osm_enabled', False),
        use_pc_relation_attention=ra_cfg.get('pc_enabled', False),
        nlir_hidden_dim=ra_cfg.get('hidden_dim', 64),
        nlir_neighbor_radius=ra_cfg.get('neighbor_radius', 5.0),
        nlir_K_safe=ra_cfg.get('K_safe', 48),
        nlir_lambda_init=ra_cfg.get('lambda_init', 0.005),
        nlir_lambda_max=ra_cfg.get('lambda_max', 0.03))
    model = model.to(config['training']['device'])
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"Model: {total / 1e6:.2f}M params ({trainable / 1e6:.2f}M trainable)")
    return model


def build_optimizer_and_scheduler(model, config, log=None):
    """Create AdamW optimizer with separate boost param group."""
    _log = log.info if log else logger.info
    t_cfg = config['training']
    lr = t_cfg['lr']
    wd = t_cfg['weight_decay']

    boost_params, other_params = [], []
    for name, param in model.named_parameters():
        if 'saliency_boost_raw' in name or 'attention_gate_raw' in name:
            boost_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': other_params, 'weight_decay': wd},
        {'params': boost_params, 'lr': lr * 2, 'weight_decay': 0.001},
    ], lr=lr)

    scheduler = None
    if t_cfg.get('use_scheduler', True):
        warmup_epochs = t_cfg['warmup_epochs']
        eta_min = t_cfg['eta_min']
        cosine_T = t_cfg['cosine_epochs']
        num_epochs = t_cfg['num_epochs']

        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_T, eta_min=eta_min)

        hold_epochs = num_epochs - warmup_epochs - cosine_T
        if hold_epochs > 0:
            hold_factors = [eta_min / pg['initial_lr'] for pg in optimizer.param_groups]
            hold = LambdaLR(optimizer, lr_lambda=[
                (lambda _step, _f=f: _f) for f in hold_factors])
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine, hold],
                milestones=[warmup_epochs, warmup_epochs + cosine_T])
            _log(f"Scheduler: Warmup({warmup_epochs}) -> "
                 f"Cosine(T={cosine_T}, eta_min={eta_min}) -> "
                 f"Hold({hold_epochs} epochs)")
        else:
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                     milestones=[warmup_epochs])
            _log(f"Scheduler: Warmup({warmup_epochs}) -> "
                 f"Cosine(T={cosine_T}, eta_min={eta_min})")

    return optimizer, scheduler


def build_dataloaders(config, log=None):
    """Create train/val DataLoaders with full determinism."""
    _log = log.info if log else logger.info
    t_cfg = config['training']
    sfaa = config['structural_fingerprint_aware_attention']
    seed = t_cfg['seed']
    num_workers = t_cfg['num_workers']
    batch_size = t_cfg['batch_size']

    train_dataset = PcMapLocDataset(
        config, mode='train',
        use_osm_saliency=sfaa['osm']['enabled'],
        use_pc_saliency=sfaa['pc']['enabled'])
    val_dataset = PcMapLocDataset(
        config, mode='val',
        use_osm_saliency=sfaa['osm']['enabled'],
        use_pc_saliency=sfaa['pc']['enabled'])

    g_train = torch.Generator()
    g_train.manual_seed(seed)
    g_val = torch.Generator()
    g_val.manual_seed(seed + 1)

    common = dict(
        collate_fn=collate_fn_BEV, pin_memory=True,
        worker_init_fn=worker_init_fn, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=True, generator=g_train, **common)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False, generator=g_val, **common)

    _log(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    return train_loader, val_loader, train_dataset


# =============================================================================
# Entry Point
# =============================================================================

def parse_args():
    """CLI: config file path only. All training params maintained in kitti.yaml."""
    parser = argparse.ArgumentParser(description='SFPR Training')
    parser.add_argument('--config', type=str, default='conf/data/kitti.yaml',
                        help='Path to config yaml file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Migrate legacy config keys to paper-aligned names
    config = normalize_config_keys(config)

    set_seed(config['training']['seed'])

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info(f"\n{'=' * 70}")
    logger.info(f"SFPR Training  |  device={config['training']['device']}  "
                f"bs={config['training']['batch_size']}  "
                f"epochs={config['training']['num_epochs']}")
    logger.info(f"Config: {args.config}")
    logger.info(f"{'=' * 70}\n")

    model = build_model(config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    train_loader, val_loader, train_dataset = build_dataloaders(config)
    loss_cfg = config['loss']
    criterion = FingerprintConsistencyLoss(
        gamma=loss_cfg['fccl_gamma'],
        m=loss_cfg['fccl_margin'],
        beta_p=loss_cfg.get('beta_p', 2.0),
        eta_p=loss_cfg.get('eta_p', 0.04),
        beta_n=loss_cfg.get('beta_n', 2.0),
        rho=loss_cfg.get('rho', 0.6),
        eta_n=loss_cfg.get('eta_n', 0.04))

    train_model(
        model, train_loader, val_loader, criterion,
        optimizer, scheduler, config,
        train_dataset=train_dataset)