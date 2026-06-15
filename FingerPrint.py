from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import igraph as ig  # type: ignore
    _HAVE_IGRAPH = True
except Exception:
    ig = None  # type: ignore
    _HAVE_IGRAPH = False

# ═══════════════════════════════════════════════════════════════════════
# Vectorised type system (replaces per-pair _sem_compat calls)
# ═══════════════════════════════════════════════════════════════════════
_TYPE_TO_ID: Dict[str, int] = {
    'building_corner': 0, 'pole': 1, 'road_fork': 2,
    'traffic_sign': 3, 'trunk': 4,
}
_N_TYPES = 5
_COMPAT_MAT = np.zeros((_N_TYPES, _N_TYPES), dtype=np.float32)
np.fill_diagonal(_COMPAT_MAT, 1.0)
_COMPAT_MAT[4, 1] = _COMPAT_MAT[1, 4] = 0.40   # trunk ↔ pole
_COMPAT_MAT[1, 3] = _COMPAT_MAT[3, 1] = 0.25   # pole ↔ traffic_sign


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def _xy_to_numpy(v: Any) -> np.ndarray:
    """Ensure [N, 2] float32 contiguous array from any input."""
    if v is None:
        return np.zeros((0, 2), dtype=np.float32)
    if hasattr(v, 'detach'):
        v = v.detach().cpu().numpy()
    arr = np.asarray(v, dtype=np.float32)
    return arr[:, :2] if arr.ndim == 2 else arr.reshape(-1, 2)

# ═══════════════════════════════════════════════════════════════════════
# Semantic compatibility (cross-modal type matching)
# ═══════════════════════════════════════════════════════════════════════

def _sem_compat(a: str, b: str) -> float:
    """Scalar semantic compatibility via _COMPAT_MAT lookup."""
    if a == b:
        return 1.0
    ai, bi = _TYPE_TO_ID.get(a, -1), _TYPE_TO_ID.get(b, -1)
    if ai < 0 or bi < 0:
        return 0.0
    return float(_COMPAT_MAT[ai, bi])

def _cosine_sim(a, b) -> float:
    if a is None or b is None:
        return 1.0
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape[0] == 0 or a.shape != b.shape:
        return 1.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ═══════════════════════════════════════════════════════════════════════
# Hypothesis — beam state: assignment set + pose T = (R, t)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FingerprintHypothesis:
    """One beam hypothesis: matched pairs + current pose T = (R, t) ∈ SE(2).

    Uses list accumulation for O(1) amortised add (replaces O(n) np.vstack).
    """
    matched_pairs: List[Tuple[int, int]] = field(default_factory=list)
    score: float = 0.0
    _q_list: List[np.ndarray] = field(default_factory=list, repr=False)
    _m_list: List[np.ndarray] = field(default_factory=list, repr=False)
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None

    def copy(self) -> "FingerprintHypothesis":
        return FingerprintHypothesis(
            list(self.matched_pairs), self.score,
            list(self._q_list), list(self._m_list),
            None if self.R is None else self.R.copy(),
            None if self.t is None else self.t.copy(),
        )

    def add(self, qi: int, mi: int, qp: np.ndarray, mp: np.ndarray, w: float):
        self.matched_pairs.append((qi, mi))
        self._q_list.append(np.asarray(qp, dtype=np.float32).ravel()[:2].copy())
        self._m_list.append(np.asarray(mp, dtype=np.float32).ravel()[:2].copy())
        self.score += w

    @property
    def query_pts(self) -> np.ndarray:
        if not self._q_list:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(self._q_list, dtype=np.float32)

    @property
    def map_pts(self) -> np.ndarray:
        if not self._m_list:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(self._m_list, dtype=np.float32)

    @staticmethod
    def from_arrays(matched_pairs, score, q_arr, m_arr, R, t):
        """Construct from pre-built arrays (used by alternating refinement)."""
        h = FingerprintHypothesis(matched_pairs=matched_pairs, score=score, R=R, t=t)
        h._q_list = [q_arr[i].copy() for i in range(q_arr.shape[0])]
        h._m_list = [m_arr[i].copy() for i in range(m_arr.shape[0])]
        return h

    def used_q(self) -> Set[int]: return {q for q, _ in self.matched_pairs}
    def used_m(self) -> Set[int]: return {m for _, m in self.matched_pairs}
    def size(self) -> int: return len(self.matched_pairs)

    def refit_pose(self):
        """Recompute pose via unweighted Kabsch."""
        if self.size() >= 2:
            self.R, self.t = solve_weighted_kabsch(self.query_pts, self.map_pts)


# ═══════════════════════════════════════════════════════════════════════
# FingerprintCandidatePool — built once per verify(), reused by all iterations
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FingerprintCandidatePool:
    r"""Candidate fingerprint element correspondence pool indexed by pair p ∈ {0,…,P−1}.

    Attributes (paper notation):
        q_xy  [P,2]  query fingerprint element positions  x_i^p
        m_xy  [P,2]  map fingerprint element positions    x_j^o
        q_id  [P]    query element IDs
        m_id  [P]    map element IDs
        omega [P]    fingerprint-aware matching weight w_ij = (1+λ·c_i) · (0.5+0.5·sim)
    """
    q_xy: np.ndarray
    m_xy: np.ndarray
    q_id: np.ndarray
    m_id: np.ndarray
    omega: np.ndarray

    @property
    def P(self) -> int:
        return self.q_xy.shape[0]

    @staticmethod
    def empty() -> "FingerprintCandidatePool":
        z2 = np.zeros((0, 2), dtype=np.float32)
        return FingerprintCandidatePool(z2, z2.copy(), np.zeros(0, dtype=np.int64),
                             np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))

# ═══════════════════════════════════════════════════════════════════════
# AnchorResult
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnchorResult:
    hypotheses: List[FingerprintHypothesis] = field(default_factory=list)
    used_query_rings: Set[int] = field(default_factory=set)
    used_active_rings: Set[int] = field(default_factory=set)


# ═══════════════════════════════════════════════════════════════════════
# SE2FingerprintVerifier  —  SG-HGV
# ═══════════════════════════════════════════════════════════════════════

class SE2FingerprintVerifier:

    def __init__(
        self,
        # ── KEEP: externally configurable ──
        verify_thresh: float = 2.5,
        graph_thresh: float = 1.5,
        lambda_attn: float = 0.5,
        beam_width: int = 20,
    ) -> None:
        # ── Primary: τ ──
        self.tau = float(verify_thresh)
        self.verify_thresh = self.tau  # alias for external callers

        # ── BIND: derived from τ ──
        self.tau_sq = self.tau ** 2
        self.radius_tol = 1.2 * self.tau
        self.trust_rmse = 0.75 * self.tau
        self.sigma_geo = 0.5 * self.tau
        self.sigma_rmse = 0.5 * self.tau
        self.anchor_max_rmse = 2.0 * self.tau
        self.search_radius_0 = 0.8 * self.tau

        # ── KEEP ──
        self.graph_thresh = float(graph_thresh)
        self.lam = float(lambda_attn)
        self.beam_width = int(beam_width)

        # ── FIX: internal constants ──
        self.max_inner_iters = 3
        self.min_active_pairs = 3
        self.pose_tol = 0.01
        self.max_branches = 5
        self.skip_penalty = 0.98
        self.max_anchor_hyps = 3
        self.anchor_min_clique = 3
        self.anchor_min_support = 2
        self.ring_tol = 2
        self.aess = (4, 4, 8, 4)  # (tau_q, tau_m, tau_pair, tau_shared)
        self.sem_thresh = 0.3
        self.min_pool_compat = 12
        self.compat_discount = 0.5
        self.collin_thresh = 0.1
        self.trust_min_size = 4
        self.trust_min_ratio = 0.30
        self.min_search_radius = 0.5
        self.radius_decay = 0.95

    # ──────────────────────────────────────────────────────────────
    # ω_ij = (1 + λ · S_ring) · (0.5 + 0.5 · sim)
    # ──────────────────────────────────────────────────────────────

    def _omega(self, saliency: float, sim: float) -> float:
        return (1.0 + self.lam * saliency) * (0.5 + 0.5 * sim)

    # ──────────────────────────────────────────────────────────────
    # Ring bucketing
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _is_collinear(pts, thresh=0.1):
        if pts.shape[0] < 3:
            return True
        c = pts - pts.mean(0, keepdims=True)
        ev = np.linalg.eigvalsh(c.T @ c / max(pts.shape[0], 1))
        return ev[1] < 1e-6 or (np.sqrt(max(ev[0], 0)) / np.sqrt(ev[1])) < thresh

    # ──────────────────────────────────────────────────────────────
    # Build candidate pool U (once per verify)
    # ──────────────────────────────────────────────────────────────

    def _build_fingerprint_candidates(self, pc_bk, osm_bk, attn, NR):
        r"""Build candidate fingerprint element correspondence set U (§4.3.1) — vectorised.

        Enumerates (i, j) satisfying type compatibility, radial proximity, and
        semantic gates. Caches ω_ij so the gate step never recomputes weights.
        """
        bw = 50.0 / NR

        pf = self._flatten_buckets(pc_bk, NR)
        mf = self._flatten_buckets(osm_bk, NR)
        if pf is None or mf is None:
            return FingerprintCandidatePool.empty()

        def _collect_vec(cross_ok):
            if cross_ok:
                valid = (compat > 0) & rho_ok
            else:
                valid = (compat > 0) & rho_ok & same_type

            # sim = cosine_sim * compat;  features are None → cosine = 1.0
            sim = compat
            valid &= (sim >= self.sem_thresh)

            qi_idx, mi_idx = np.where(valid)
            if qi_idx.shape[0] == 0:
                return None

            return FingerprintCandidatePool(
                q_xy[qi_idx].astype(np.float32),
                m_xy[mi_idx].astype(np.float32),
                q_gid[qi_idx].astype(np.int64),
                m_gid[mi_idx].astype(np.int64),
                omega.astype(np.float32),
            )

        pool = _collect_vec(False)
        if pool is None or pool.P < self.min_pool_compat:
            pool = _collect_vec(True)
        return pool if pool is not None else FingerprintCandidatePool.empty()

    # ──────────────────────────────────────────────────────────────
    # Phase 1: structural anchoring (clique)
    # ──────────────────────────────────────────────────────────────

    def _initialize_fingerprint_alignment(self, pc_bk, osm_bk, attn, ring_order, NR):
        r"""Phase 1: progressive AESS + quality-gated clique anchoring."""
        geo_sig = 0.5
        prefix: List[int] = []
        active: Set[int] = set()
        last_nc = -1
        aQ, aM, aP, aS = self.aess

            if _HAVE_IGRAPH:
                g = ig.Graph(n=M, edges=edges.tolist(), directed=False)
                g.es["weight"] = ew.tolist()
                cliques = g.maximal_cliques(min=self.anchor_min_clique)
            else:
                adj_s: List[Set[int]] = [set() for _ in range(M)]
                for u, v in edges.tolist():
                    adj_s[u].add(v); adj_s[v].add(u)
                cliques = _bron_kerbosch(adj_s, self.anchor_min_clique)

            if not cliques:
                continue


            if anchors:
                anchors.sort(key=lambda x: x.score, reverse=True)
                return AnchorResult(
                    anchors[:self.max_anchor_hyps],
                    set(prefix), set(active),
                )
        return AnchorResult()

    # ──────────────────────────────────────────────────────────────
    # Phase 2: beam search
    # ──────────────────────────────────────────────────────────────

    def _progressive_branch(self, hyp, q_nodes, m_nodes, sal, sr):
        """Pose-conditioned spatial gate: project q through T, match nearby m."""
        if hyp.R is None or hyp.t is None:
            hyp.refit_pose()
            if hyp.R is None:
                sk = hyp.copy(); sk.score *= self.skip_penalty; return [sk]
      
        return branches

    def _prune(self, hyps, jt=0.8):
        if len(hyps) <= 1:
            return hyps
        hyps = sorted(hyps, key=lambda h: h.score, reverse=True)
        kept, ks = [], []
        for h in hyps:
            s = set(h.matched_pairs)
            if all((len(s & k) / max(len(s | k), 1)) <= jt for k in ks):
                kept.append(h); ks.append(s)
        return kept

    # ──────────────────────────────────────────────────────────────
    # Phase 3: bounded Gate/SVD alternating refinement (§3–4)
    # ──────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────
    # Phase 4: final scoring (strict 1-to-1)
    # ──────────────────────────────────────────────────────────────

    def _extract_fingerprint_score(self, hyp, pool):
        _e = {"n_inliers": 0, "rmse": float("inf"), "inlier_ratio": 0.0,
              "mean_pair_weight": 0.0, "geo_score": 0.0, "clique_size": hyp.size(),
              "R": None, "t": None, "theta": None, "rho": None,
              "is_valid": False, "unique_pairs": 0,
              "coverage_query": 0.0, "coverage_map": 0.0}

        if hyp.size() < 2 or pool.P == 0:
            return _e
        if hyp.R is None or hyp.t is None:
            hyp.refit_pose()
        if hyp.R is None:
            return _e

        geo = float(ir * (1.0 / (1.0 + rmse / self.sigma_rmse)) * mw)

        return {
            "n_inliers": nU, "rmse": rmse, "inlier_ratio": ir,
            "mean_pair_weight": mw, "geo_score": geo,
            "clique_size": hyp.size(), "unique_pairs": nU,
            "coverage_query": float(len(set(pool.q_id[assigned].tolist()))) / npq,
            "coverage_map": float(len(set(pool.m_id[assigned].tolist()))) / npm,
            "R": R, "t": t,
            "theta": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
            "rho": float(np.linalg.norm(t)),
            "is_valid": True,
        }

    # ──────────────────────────────────────────────────────────────
    # Trust decision (strict AND)
    # ──────────────────────────────────────────────────────────────

    def _evaluate_fingerprint_trust(self, best, pool):
        N_eff = float(min(
            int(np.unique(pool.q_id).shape[0]),
            int(np.unique(pool.m_id).shape[0]),
        )) if pool.P > 0 else 1.0
        min_inl = max(3, int(np.ceil(0.18 * N_eff)))
        min_rat = max(0.12, min(self.trust_min_ratio, float(min_inl) / max(N_eff, 1.0)))

        checks = {
            "clique_ok": int(best["clique_size"]) >= self.trust_min_size,
            "inliers_ok": int(best["n_inliers"]) >= min_inl,
            "ratio_ok": float(best["inlier_ratio"]) >= min_rat,
            "rmse_ok": float(best["rmse"]) <= self.trust_rmse,
            "pairs_ok": int(best["unique_pairs"]) >= self.min_active_pairs,
        }
        return all(checks.values()), checks

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def match_structural_fingerprint(
        self,
        pc_nodes, osm_nodes, pc_radii, osm_radii,
        pc_attn, osm_attn,
        pc_features=None, osm_features=None,
        r_max=50.0, num_rings=30,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        r"""Run structural fingerprint matching.

        Returns (geo_score, is_trusted, dbg).
        """
        _ = osm_attn
        NR = int(num_rings)

        pc_bk = self._bucket(pc_nodes, pc_radii, pc_features, r_max, NR)
        osm_bk = self._bucket(osm_nodes, osm_radii, osm_features, r_max, NR)

        attn = np.asarray(pc_attn, dtype=np.float32).ravel()
        if attn.shape[0] < NR:
            attn = np.pad(attn, (0, NR - attn.shape[0]))
        ring_order = list(np.argsort(attn)[::-1])

        return self._run_matching_phases(pc_bk, osm_bk, attn, ring_order, NR)

    # ──────────────────────────────────────────────────────────────
    # Public API (cached PC bucketing variant)
    # ──────────────────────────────────────────────────────────────

    def match_with_query_cache(
        self,
        pc_bk: Dict,
        pc_attn,
        osm_nodes, osm_radii, osm_attn,
        osm_features=None,
        r_max: float = 50.0,
        num_rings: int = 30,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        r"""match_structural_fingerprint() variant with pre-computed PC bucketing.

        When the same query is verified against M candidates, the caller
        computes pc_bk = _bucket(pc_nodes, pc_radii, ...) once and passes
        it here for each candidate.  Saves ~(M-1)/M of PC bucketing cost.
        """
        _ = osm_attn
        NR = int(num_rings)
        osm_bk = self._bucket(osm_nodes, osm_radii, osm_features, r_max, NR)

        attn = np.asarray(pc_attn, dtype=np.float32).ravel()
        if attn.shape[0] < NR:
            attn = np.pad(attn, (0, NR - attn.shape[0]))
        ring_order = list(np.argsort(attn)[::-1])

        return self._run_matching_phases(pc_bk, osm_bk, attn, ring_order, NR)

    # ──────────────────────────────────────────────────────────────
    # Shared Phase 1–4 core
    # ──────────────────────────────────────────────────────────────

    def _run_matching_phases(self, pc_bk, osm_bk, attn, ring_order, NR):
        """Core verification pipeline shared by match_structural_fingerprint() and match_with_query_cache()."""
        tQ = sum(len(v) for v in pc_bk.values())
        tM = sum(len(v) for v in osm_bk.values())

        if tQ < 3 or tM < 3:
            return 0.0, False, {"verified": True, "phase": "early_exit",
                                "total_query_nodes": tQ, "total_map_nodes": tM,
                                "reason": "insufficient_nodes"}

        pool = self._build_fingerprint_candidates(pc_bk, osm_bk, attn, NR)

        p1 = self._initialize_fingerprint_alignment(pc_bk, osm_bk, attn, ring_order, NR)
        if not p1.hypotheses:
            return 0.0, False, {"verified": True, "phase": "phase1_failed",
                                "total_query_nodes": tQ, "total_map_nodes": tM,
                                "reason": "no_anchor"}

        # Phase 2: beam search
        beam = p1.hypotheses
        remaining = [r for r in ring_order if r not in p1.used_active_rings]
        sr = self.search_radius_0
        best_sz = max(h.size() for h in beam)

        for ri in remaining:
            if not beam:
                break
            qn = pc_bk.get(ri, [])
            if not qn:
                continue
            mn: List[Dict] = []
            for dr in range(-self.ring_tol, self.ring_tol + 1):
                nb = ri + dr
                if 0 <= nb < NR:
                    mn.extend(osm_bk.get(nb, []))
            if not mn:
                continue
            sal = float(attn[ri]) if ri < len(attn) else 0.0
            nb_list: List[FingerprintHypothesis] = []
            for hyp in beam:
                nb_list.extend(self._progressive_branch(hyp, qn, mn, sal, sr))
            nb_list = self._prune(nb_list)
            nb_list.sort(key=lambda h: h.score, reverse=True)
            beam = nb_list[:self.beam_width]
            ns = max((h.size() for h in beam), default=0)
            if ns > best_sz:
                sr = max(self.min_search_radius, sr * self.radius_decay)
                best_sz = ns

        if not beam:
            return 0.0, False, {"verified": True, "phase": "phase2_exhausted",
                                "total_query_nodes": tQ, "total_map_nodes": tM,
                                "reason": "beam_exhausted"}

        # Phase 3: Gate/SVD refinement
        rdbg_best: Dict[str, Any] = {}
        if pool.P > 0:
            ref = []
            for hyp in beam:
                rh, rd = self._alternating_fingerprint_refinement(hyp, pool)
                ref.append(rh)
                if not rdbg_best or rh.score > rdbg_best.get("_s", -1e30):
                    rdbg_best = rd; rdbg_best["_s"] = rh.score
            ref = self._prune(ref)
            ref.sort(key=lambda h: h.score, reverse=True)
            beam = ref[:self.beam_width]
        rdbg_best.pop("_s", None)

        # Phase 4: final scoring
        results = [self._extract_fingerprint_score(h, pool) for h in beam if h.size() >= 3]
        if not results:
            return 0.0, False, {"verified": True, "phase": "scoring_failed",
                                "total_query_nodes": tQ, "total_map_nodes": tM,
                                "reason": "no_valid_hypotheses"}

        results.sort(key=lambda r: r["geo_score"] if r["is_valid"] else -1e30, reverse=True)
        best = results[0]
        geo = float(best["geo_score"])
        trusted, checks = self._evaluate_fingerprint_trust(best, pool)

        dbg: Dict[str, Any] = {
            "verified": True, "phase": "complete",
            "total_query_nodes": tQ, "total_map_nodes": tM,
            "num_anchors": len(p1.hypotheses),
            "final_beam_size": len(beam),
            "best_clique_size": int(best["clique_size"]),
            "best_unique_pairs": int(best["unique_pairs"]),
            "best_inliers": int(best["n_inliers"]),
            "best_rmse": float(best["rmse"]),
            "best_inlier_ratio": float(best["inlier_ratio"]),
            "best_mean_pair_weight": float(best["mean_pair_weight"]),
            "geo_score": geo,
            "is_trusted": trusted,
            "trust_checks": checks,
            "candidate_pool_size": pool.P,
            "used_fallback_clique": not _HAVE_IGRAPH,
        }
        if rdbg_best:
            dbg["inner_iters"] = rdbg_best.get("inner_iters", 0)
            dbg["stopped_by"] = rdbg_best.get("stopped_by", "n/a")
        if best.get("theta") is not None:
            dbg["best_pose_theta"] = float(best["theta"])
            dbg["best_pose_rho"] = float(best["rho"])

        return geo, trusted, dbg
