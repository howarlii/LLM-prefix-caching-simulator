"""Offline-optimal ILP solver for the prefix-cache scheduling problem.

Given the full request sequence, this module computes the **provably
optimal** ``saved_time_rate`` (within the model below) by formulating
the cache scheduling problem as a 0/1 mixed integer linear program and
solving it with PuLP/CBC.

The ILP is intended for **small-to-medium instances** (≤ ~500 requests
on a typical workload).  It is meant as a ground-truth benchmark for
the greedy oracle (``OracleGreedyStrategy``) and the online strategies.

Model
=====
* The simulator's per-request saved time
  (see ``src/metrics.py`` and ``ModelConfig.incremental_prefill_flop``)
  is **linear in** ``hit_tokens`` for a fixed total length:

      compute_saved(T, h) = h * c(T) / gpu_flops

  where ``c(T) = prefill_flop(T) / T``.  This linearity is exploited
  to make the objective linear in the hit-depth indicator variables.

* DRAM is a **superset** of HBM: any node currently in HBM is also in
  DRAM (mirror).  Mamba state can be created only when the node is in
  HBM (the simulator only sets mamba state on newly inserted HBM nodes
  or on existing HBM nodes during request processing); once created it
  may persist as long as the node remains in DRAM.

* Cache contents only change at request processing time.  Between two
  requests cache size is monotonically non-increasing (only evictions
  happen), so capacity needs to be enforced only **after each request**.

Variables
---------
For each global radix node ``n`` and each timestep ``t`` (1-indexed,
``t`` = "state after processing request ``t``"):

* ``hbm[n,t]``  ∈ {0,1} — node ``n`` is in HBM after step ``t``.
* ``dram[n,t]`` ∈ {0,1} — node ``n`` is in DRAM after step ``t``.
* ``mamba[n,t]`` ∈ {0,1} — usable mamba state for ``n`` exists somewhere
  in cache after step ``t``.
* ``hm[n,t]`` = ``hbm[n,t] ∧ mamba[n,t]`` — auxiliary AND for capacity
  (only HBM-resident mamba states cost HBM bytes).
* For each request ``r`` and each node ``n_i`` on its global path:
  ``u_h[r,i], u_t[r,i]`` — "this node is the deepest HBM-only / total
  hit for request r".  At most one ``u_h`` and one ``u_t`` per request.

Variables are only created for nodes visited ≥ 2 times in the global
tree, and only for timesteps inside the node's "alive interval"
``[t_first[n], t_last[n]]``.  Single-use nodes are never worth caching.

Objective
---------
Maximise the total saved time:

    ∑_r [(a[r] - b) * t_depth[r] + b * h_depth[r]]

where:
* ``a[r] = prefill_flop(T_r) / T_r / gpu_flops``  — per-token compute
  saving rate for request ``r``.
* ``b = kv_bytes_per_token / pcie_bandwidth``     — per-token PCIe cost.
* ``t_depth[r] = ∑_i depth(n_i) * u_t[r,i]``       — total hit depth.
* ``h_depth[r] = ∑_i depth(n_i) * u_h[r,i]``       — HBM-only hit depth.

A side constraint ``t_depth[r] ≥ h_depth[r]`` enforces "DRAM extension
extends HBM, never replaces it".  (Both terms are then linear in the
indicator vars.)

Constraints
-----------
1. DRAM ⊇ HBM:                       d[n,t] ≥ h[n,t]
2. Mamba in some tier:               m[n,t] ≤ d[n,t]
3. Mamba creation only on visit:     m[n,t] - m[n,t-1] ≤ on_path[n,t]
4. Cached transitions on visit:      h[n,t] - h[n,t-1] ≤ on_path[n,t]
                                     d[n,t] - d[n,t-1] ≤ on_path[n,t]
5. Tree structure:                   h[n,t] ≤ h[parent(n),t]
                                     d[n,t] ≤ d[parent(n),t]
6. AND linearisation:                hm ≤ h, hm ≤ m, hm ≥ h+m-1
7. Capacity HBM:  ∑_n tokens(n)*h[n,t] + ∑_n M*hm[n,t] ≤ C_HBM
8. Capacity DRAM: ∑_n tokens(n)*d[n,t] + ∑_n M*m[n,t]  ≤ C_DRAM
9. Hit indicators feasibility:
       u_h[r,i] ≤ h[n_i,r-1]; u_h[r,i] ≤ m[n_i,r-1]; ∑_i u_h[r,i] ≤ 1
       u_t[r,i] ≤ d[n_i,r-1]; u_t[r,i] ≤ m[n_i,r-1]; ∑_i u_t[r,i] ≤ 1
10. ``∑_i depth(n_i) u_t[r,i] ≥ ∑_i depth(n_i) u_h[r,i]``

Tractability
============
For 100–200 requests on a typical loogle-style workload the LP has
~10⁵ binary variables and CBC solves it in minutes.  500 requests is
borderline — expect tens of minutes to hours, and consider passing a
``time_limit_s`` to fall back to the best feasible incumbent.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.model_config import DEFAULT_MODEL, ModelConfig
from src.radix_tree import PageKey, RadixNode, RadixTree


def _tokens_to_pages(tokens: List[int], page_size: int) -> List[PageKey]:
    """Local copy of cache_simulator.tokens_to_pages to avoid a circular import."""
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    pages: List[PageKey] = []
    for i in range(0, len(tokens), page_size):
        pages.append(tuple(tokens[i : i + page_size]))
    return pages


@dataclass
class OracleILPResult:
    """Result of an ILP solve."""

    status: str
    saved_time_rate: float
    total_saved_time: float
    gpu_compute_time_no_cache: float
    gpu_compute_time_with_cache: float
    pcie_total_transfer_bytes: int
    pcie_total_transfer_time: float
    flop_save_rate: float
    total_flop_no_cache: float
    total_flop_with_cache: float
    hbm_token_hit_rate: float
    dram_token_hit_rate: float
    per_request_hit_tokens: List[int]
    per_request_dram_hit_tokens: List[int]
    per_request_input_tokens: List[int]
    per_request_saved_time: List[float]
    num_binary_vars: int
    num_constraints: int
    solve_time_s: float

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "saved_time_rate": self.saved_time_rate,
            "total_saved_time": self.total_saved_time,
            "gpu_compute_time_no_cache": self.gpu_compute_time_no_cache,
            "gpu_compute_time_with_cache": self.gpu_compute_time_with_cache,
            "pcie_total_transfer_bytes": self.pcie_total_transfer_bytes,
            "pcie_total_transfer_time": self.pcie_total_transfer_time,
            "flop_save_rate": self.flop_save_rate,
            "total_flop_no_cache": self.total_flop_no_cache,
            "total_flop_with_cache": self.total_flop_with_cache,
            "hbm_token_hit_rate": self.hbm_token_hit_rate,
            "dram_token_hit_rate": self.dram_token_hit_rate,
            "num_binary_vars": self.num_binary_vars,
            "num_constraints": self.num_constraints,
            "solve_time_s": self.solve_time_s,
        }

    def to_run_metrics_dict(self) -> Dict:
        """Shape the result as a ``RunMetrics.to_dict()``-compatible payload
        so it can flow through ``persist_result_row`` unchanged.

        Fields that don't apply to an ILP (tree shape histograms, cache
        usage samples, branch-rate counters, DRAM-tier flow) are filled
        with zeros / empty dicts — consistent with a single-tier HBM-only
        run where nothing was tracked at timestep granularity.
        """
        import math

        def _percentile(sv, p):
            if not sv:
                return 0.0
            if p <= 0:
                return sv[0]
            if p >= 100:
                return sv[-1]
            k = (len(sv) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sv[int(k)]
            return sv[f] + (sv[c] - sv[f]) * (k - f)

        n_req = len(self.per_request_input_tokens)
        total_in = sum(self.per_request_input_tokens)
        load_tokens = sum(self.per_request_hit_tokens)
        compute_tokens = total_in - load_tokens
        lcr = (load_tokens / compute_tokens) if compute_tokens else None

        pst = sorted(self.per_request_saved_time)
        pst_mean = (sum(pst) / len(pst)) if pst else 0.0

        return {
            # Tier hit rates
            "hbm_token_hit_rate": self.hbm_token_hit_rate,
            "dram_token_hit_rate": self.dram_token_hit_rate,
            # Per-request saved-time distribution
            "per_request_saved_time_mean": pst_mean,
            "per_request_saved_time_p50": _percentile(pst, 50),
            "per_request_saved_time_p90": _percentile(pst, 90),
            "per_request_saved_time_p99": _percentile(pst, 99),
            # Token-level summary
            "load_tokens": load_tokens,
            "compute_tokens": compute_tokens,
            "load_compute_ratio": lcr,
            "peak_cached_tokens": 0,        # not tracked by the ILP
            "avg_cached_tokens": 0.0,       # not tracked by the ILP
            "num_requests": n_req,
            "total_input_tokens": total_in,
            # Tree-shape histograms: not produced by the ILP
            "tree_depth_histogram": {},
            "valid_cached_depth_histogram": {},
            "tree_access_by_depth": {},
            "access_percentiles_by_depth": {},
            # Branch stats: not produced by the ILP
            "req_branch_rate": 0.0,
            "req_new_branch_rate": 0.0,
            # FLOP + time
            "total_flop_no_cache": self.total_flop_no_cache,
            "total_flop_with_cache": self.total_flop_with_cache,
            "flop_save_rate": self.flop_save_rate,
            "gpu_compute_time_no_cache": self.gpu_compute_time_no_cache,
            "gpu_compute_time_with_cache": self.gpu_compute_time_with_cache,
            "pcie_total_transfer_bytes": self.pcie_total_transfer_bytes,
            "pcie_total_transfer_time": self.pcie_total_transfer_time,
            "total_saved_time": self.total_saved_time,
            "saved_time_rate": self.saved_time_rate,
            # DRAM fields — always zero (ILP here is HBM-only)
            "dram_peak_cached_tokens": 0,
            "dram_avg_cached_tokens": 0.0,
            "avg_promoted_tokens_per_req": 0.0,
            "avg_promoted_nodes_per_req": 0.0,
            "avg_promoted_gb_per_req": None,
            "total_promoted_tokens": 0,
            "total_promoted_nodes": 0,
            "avg_demoted_tokens_per_req": 0.0,
            "avg_demoted_nodes_per_req": 0.0,
            "avg_demoted_gb_per_req": None,
            "total_demoted_tokens": 0,
            "total_demoted_nodes": 0,
        }


class OracleILPSolver:
    """Build and solve the offline-optimal cache scheduling ILP."""

    def __init__(
        self,
        requests_token_ids: Sequence[Sequence[int]],
        page_size: int,
        capacity_hbm_tokens: Optional[int],
        *,
        dram_capacity_tokens: Optional[int] = 0,
        model: ModelConfig = DEFAULT_MODEL,
        gpu_flops: float,
        pcie_bandwidth: float,
    ) -> None:
        if capacity_hbm_tokens is None:
            raise ValueError("ILP requires a finite HBM capacity")
        if gpu_flops <= 0:
            raise ValueError("gpu_flops must be positive")

        self.model = model
        self.page_size = page_size
        self.gpu_flops = gpu_flops
        self.pcie_bandwidth = pcie_bandwidth
        self.C_hbm = int(capacity_hbm_tokens)
        self.C_dram = int(dram_capacity_tokens) if dram_capacity_tokens else 0
        self.dram_enabled = self.C_dram > 0
        self.M = model.mamba_state_token_equiv
        self.b_per_token = (
            model.kv_bytes_per_token / pcie_bandwidth
            if pcie_bandwidth and pcie_bandwidth > 0
            else 0.0
        )

        # ----- Page-tokenise every request -----
        # The simulator always drops a trailing partial page; mirror that.
        self.req_pages = []
        for tids in requests_token_ids:
            pgs = _tokens_to_pages(list(tids), page_size)
            if len(pgs) > 1 and len(pgs[-1]) < page_size:
                pgs = pgs[:-1]
            self.req_pages.append(pgs)
        self.T = len(self.req_pages)
        self.req_total_tokens: List[int] = [
            sum(len(p) for p in pgs) for pgs in self.req_pages
        ]
        # a[r] = per-token compute saving for request r (seconds per token).
        # Derivation: saved(h) = (prefill(T) - incr(T,h)) / F  is linear in h
        # with slope c(T)/F where c(T) = prefill(T) / T (proven by direct
        # expansion of the FLOP formulas in model_config.py).
        self.a: List[float] = [
            (model.prefill_flop(tt) / tt / gpu_flops) if tt > 0 else 0.0
            for tt in self.req_total_tokens
        ]

        # ----- Build the global radix tree -----
        self.global_tree = RadixTree(model=model)
        for pgs in self.req_pages:
            self.global_tree.simulate_request(pgs)

        # ----- Walk each request through the *final* global tree to collect:
        #       - integer node ids (so the LP can index them)
        #       - per-request path = list of (node_idx, depth_tokens)
        #       - per-node sorted list of visiting request indices
        # ----------------------------------------------------------------
        self._node_id_of: Dict[int, int] = {}
        self.node_list: List[RadixNode] = []
        self.node_depths: List[int] = []
        self.node_tokens: List[int] = []
        self.node_parent: List[int] = []
        self.node_visits: Dict[int, List[int]] = defaultdict(list)
        self.req_paths: List[List[Tuple[int, int]]] = []

        def _ensure_node_id(n: RadixNode) -> int:
            nid = self._node_id_of.get(id(n))
            if nid is not None:
                return nid
            # Resolve parent first.
            if n.parent is None or n.parent is self.global_tree.root:
                parent_id = -1
            else:
                parent_id = _ensure_node_id(n.parent)
            nid = len(self.node_list)
            self._node_id_of[id(n)] = nid
            self.node_list.append(n)
            self.node_depths.append(n.depth_tokens)
            self.node_tokens.append(n.num_tokens)
            self.node_parent.append(parent_id)
            return nid

        for ri, pgs in enumerate(self.req_pages):
            node = self.global_tree.root
            pi = 0
            path: List[Tuple[int, int]] = []
            while pi < len(pgs):
                ch = node.children.get(pgs[pi])
                if ch is None:
                    break
                ok = True
                for j in range(len(ch.pages)):
                    if pi + j >= len(pgs) or pgs[pi + j] != ch.pages[j]:
                        ok = False
                        break
                if not ok:
                    break
                nid = _ensure_node_id(ch)
                self.node_visits[nid].append(ri)
                path.append((nid, ch.depth_tokens))
                pi += len(ch.pages)
                node = ch
            self.req_paths.append(path)

        self.N = len(self.node_list)

        # ----- Active set: nodes visited >= 2 times. ----------------------
        # Visited-once nodes can never produce a future hit, so caching them
        # is always wasteful — fix their cache state to 0 by simply not
        # creating LP variables.  Internal nodes whose children are reused
        # by other requests are themselves visited by those requests
        # (transitively), so the active set is closed under "ancestor".
        self.t_first: Dict[int, int] = {}
        self.t_last: Dict[int, int] = {}
        self.active_nodes: List[int] = []
        for nidx, vis in self.node_visits.items():
            if len(vis) >= 2:
                self.t_first[nidx] = vis[0]
                self.t_last[nidx] = vis[-1]
                self.active_nodes.append(nidx)
        self.active_nodes.sort()

        # Per-node total no-cache compute time (used for the saved-time-rate).
        self.gpu_compute_no_cache = sum(
            model.prefill_flop(tt) / gpu_flops for tt in self.req_total_tokens
        )
        self.flop_no_cache = sum(
            model.prefill_flop(tt) for tt in self.req_total_tokens
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        time_limit_s: Optional[int] = None,
        msg: bool = False,
        relax: bool = False,
    ) -> OracleILPResult:
        """Build and solve the ILP.  ``relax=True`` solves the LP
        relaxation (continuous in [0,1]) for an upper bound."""
        import time

        import pulp

        prob = pulp.LpProblem("oracle_kv_cache", pulp.LpMaximize)
        cat = "Continuous" if relax else "Binary"
        var_lb, var_ub = 0.0, 1.0

        def make_var(name: str):
            return pulp.LpVariable(name, lowBound=var_lb, upBound=var_ub, cat=cat)

        # ---------- Cache state vars ----------
        h: Dict[Tuple[int, int], pulp.LpVariable] = {}
        d: Dict[Tuple[int, int], pulp.LpVariable] = {}
        m: Dict[Tuple[int, int], pulp.LpVariable] = {}
        hm: Dict[Tuple[int, int], pulp.LpVariable] = {}

        for n in self.active_nodes:
            t_lo = self.t_first[n] + 1  # 1-indexed timestep just after first visit
            t_hi = self.t_last[n] + 1
            for t in range(t_lo, t_hi + 1):
                h[(n, t)] = make_var(f"h_{n}_{t}")
                d[(n, t)] = make_var(f"d_{n}_{t}")
                m[(n, t)] = make_var(f"m_{n}_{t}")
                hm[(n, t)] = make_var(f"hm_{n}_{t}")

        # Precompute on_path[n,t]: 1 iff request t (1-indexed; equiv. ri = t-1)
        # visits node n in the global tree.
        on_path: Dict[Tuple[int, int], int] = {}
        for n in self.active_nodes:
            for ri in self.node_visits[n]:
                on_path[(n, ri + 1)] = 1

        # ---------- Per-node constraints ----------
        for n in self.active_nodes:
            t_lo = self.t_first[n] + 1
            t_hi = self.t_last[n] + 1
            parent = self.node_parent[n]
            for t in range(t_lo, t_hi + 1):
                hv, dv, mv, hmv = h[(n, t)], d[(n, t)], m[(n, t)], hm[(n, t)]
                # DRAM ⊇ HBM (when DRAM disabled, force dram ≡ hbm so the
                # LP can't "cache everything for free" in an unbounded
                # DRAM tier that doesn't actually exist).
                if self.dram_enabled:
                    prob += dv >= hv, f"hbm_le_dram_{n}_{t}"
                else:
                    prob += dv == hv, f"no_dram_{n}_{t}"
                # Mamba ≤ DRAM
                prob += mv <= dv, f"m_le_d_{n}_{t}"
                # AND linearisation: hm = h * m
                prob += hmv <= hv, f"hm_le_h_{n}_{t}"
                prob += hmv <= mv, f"hm_le_m_{n}_{t}"
                prob += hmv >= hv + mv - 1, f"hm_ge_{n}_{t}"
                # Transitions: cache + mamba can only INCREASE on a visit.
                op = on_path.get((n, t), 0)
                if t > t_lo:
                    prob += hv - h[(n, t - 1)] <= op, f"h_trans_{n}_{t}"
                    prob += dv - d[(n, t - 1)] <= op, f"d_trans_{n}_{t}"
                    prob += mv - m[(n, t - 1)] <= op, f"m_trans_{n}_{t}"
                else:
                    # First active timestep — must come from "empty" state.
                    prob += hv <= op, f"h_init_{n}_{t}"
                    prob += dv <= op, f"d_init_{n}_{t}"
                    prob += mv <= op, f"m_init_{n}_{t}"
                # Tree structure: h[n,t] ≤ h[parent,t], d similar.
                if parent >= 0 and parent in self.t_first:
                    if (parent, t) in h:
                        prob += hv <= h[(parent, t)], f"tree_h_{n}_{t}"
                        prob += dv <= d[(parent, t)], f"tree_d_{n}_{t}"

        # ---------- Capacity constraints ----------
        # Group vars by timestep so we add one capacity constraint per t.
        by_t_h: Dict[int, List[Tuple[int, pulp.LpVariable]]] = defaultdict(list)
        by_t_d: Dict[int, List[Tuple[int, pulp.LpVariable]]] = defaultdict(list)
        by_t_hm: Dict[int, List[pulp.LpVariable]] = defaultdict(list)
        by_t_m: Dict[int, List[pulp.LpVariable]] = defaultdict(list)
        for (n, t), v in h.items():
            by_t_h[t].append((self.node_tokens[n], v))
        for (n, t), v in d.items():
            by_t_d[t].append((self.node_tokens[n], v))
        for (n, t), v in hm.items():
            by_t_hm[t].append(v)
        for (n, t), v in m.items():
            by_t_m[t].append(v)

        for t in range(1, self.T + 1):
            terms: List = []
            for tok, v in by_t_h.get(t, []):
                terms.append(tok * v)
            for v in by_t_hm.get(t, []):
                terms.append(self.M * v)
            if terms:
                prob += pulp.lpSum(terms) <= self.C_hbm, f"cap_hbm_{t}"
            if self.dram_enabled:
                terms2: List = []
                for tok, v in by_t_d.get(t, []):
                    terms2.append(tok * v)
                for v in by_t_m.get(t, []):
                    terms2.append(self.M * v)
                if terms2:
                    prob += pulp.lpSum(terms2) <= self.C_dram, f"cap_dram_{t}"

        # ---------- Hit indicator vars ----------
        u_h: Dict[Tuple[int, int], pulp.LpVariable] = {}
        u_t: Dict[Tuple[int, int], pulp.LpVariable] = {}
        for r in range(self.T):
            path = self.req_paths[r]
            state_t = r  # state at end of step r (1-indexed) is "before request r+1"
            if state_t == 0:
                continue
            for i, (n_idx, depth) in enumerate(path):
                if n_idx not in self.t_first:
                    continue
                if (n_idx, state_t) not in h:
                    continue
                uh_var = make_var(f"uh_{r}_{i}")
                ut_var = make_var(f"ut_{r}_{i}")
                u_h[(r, i)] = uh_var
                u_t[(r, i)] = ut_var
                prob += uh_var <= h[(n_idx, state_t)], f"uh_h_{r}_{i}"
                prob += uh_var <= m[(n_idx, state_t)], f"uh_m_{r}_{i}"
                prob += ut_var <= d[(n_idx, state_t)], f"ut_d_{r}_{i}"
                prob += ut_var <= m[(n_idx, state_t)], f"ut_m_{r}_{i}"

        # At-most-one indicator per request, plus t_depth ≥ h_depth.
        for r in range(self.T):
            path = self.req_paths[r]
            uh_terms = []
            ut_terms = []
            uh_depth_terms = []
            ut_depth_terms = []
            for i, (_, depth) in enumerate(path):
                if (r, i) in u_h:
                    uh_terms.append(u_h[(r, i)])
                    uh_depth_terms.append(depth * u_h[(r, i)])
                if (r, i) in u_t:
                    ut_terms.append(u_t[(r, i)])
                    ut_depth_terms.append(depth * u_t[(r, i)])
            if uh_terms:
                prob += pulp.lpSum(uh_terms) <= 1, f"uh_one_{r}"
            if ut_terms:
                prob += pulp.lpSum(ut_terms) <= 1, f"ut_one_{r}"
            if ut_depth_terms or uh_depth_terms:
                prob += (
                    pulp.lpSum(ut_depth_terms) >= pulp.lpSum(uh_depth_terms)
                ), f"t_ge_h_{r}"

        # ---------- Objective ----------
        obj_terms: List = []
        for r in range(self.T):
            a_r = self.a[r]
            path = self.req_paths[r]
            for i, (_, depth) in enumerate(path):
                if (r, i) in u_t:
                    coef = (a_r - self.b_per_token) * depth
                    if coef != 0:
                        obj_terms.append(coef * u_t[(r, i)])
                if (r, i) in u_h:
                    coef = self.b_per_token * depth
                    if coef != 0:
                        obj_terms.append(coef * u_h[(r, i)])
        if not obj_terms:
            prob += 0
        else:
            prob += pulp.lpSum(obj_terms)

        num_vars = len(prob.variables())
        num_constraints = len(prob.constraints)

        t0 = time.time()
        solver_kwargs = {"msg": msg}
        if time_limit_s is not None:
            solver_kwargs["timeLimit"] = time_limit_s
        solver = pulp.PULP_CBC_CMD(**solver_kwargs)
        prob.solve(solver)
        solve_time = time.time() - t0
        status = pulp.LpStatus[prob.status]

        # ---------- Extract per-request hit depths ----------
        per_req_hit: List[int] = [0] * self.T
        per_req_dram_extra: List[int] = [0] * self.T
        for r in range(self.T):
            path = self.req_paths[r]
            best_h = 0
            best_t = 0
            for i, (_, depth) in enumerate(path):
                if (r, i) in u_h and (u_h[(r, i)].value() or 0.0) > 0.5:
                    best_h = max(best_h, depth)
                if (r, i) in u_t and (u_t[(r, i)].value() or 0.0) > 0.5:
                    best_t = max(best_t, depth)
            best_t = max(best_t, best_h)  # safety
            per_req_hit[r] = best_t
            per_req_dram_extra[r] = best_t - best_h

        # ---------- Compute final metrics ----------
        per_req_saved: List[float] = []
        flop_with_cache = 0.0
        for r in range(self.T):
            flop_no = self.model.prefill_flop(self.req_total_tokens[r])
            flop_wc = self.model.incremental_prefill_flop(
                self.req_total_tokens[r], per_req_hit[r]
            )
            flop_with_cache += flop_wc
            compute_avoided_r = (flop_no - flop_wc) / self.gpu_flops
            transfer_r = (
                per_req_dram_extra[r] * self.model.kv_bytes_per_token
                / self.pcie_bandwidth
                if self.pcie_bandwidth and self.pcie_bandwidth > 0
                else 0.0
            )
            per_req_saved.append(compute_avoided_r - transfer_r)

        flop_save = self.flop_no_cache - flop_with_cache
        gpu_with_cache = flop_with_cache / self.gpu_flops
        compute_avoided = self.gpu_compute_no_cache - gpu_with_cache
        pcie_bytes = sum(
            per_req_dram_extra[r] * self.model.kv_bytes_per_token
            for r in range(self.T)
        )
        pcie_time = (
            pcie_bytes / self.pcie_bandwidth
            if self.pcie_bandwidth and self.pcie_bandwidth > 0
            else 0.0
        )
        total_saved_time = compute_avoided - pcie_time
        saved_time_rate = (
            total_saved_time / self.gpu_compute_no_cache
            if self.gpu_compute_no_cache > 0
            else 0.0
        )

        total_input = sum(self.req_total_tokens)
        hbm_hit_total = sum(
            per_req_hit[r] - per_req_dram_extra[r] for r in range(self.T)
        )
        dram_hit_total = sum(per_req_dram_extra)

        return OracleILPResult(
            status=status,
            saved_time_rate=saved_time_rate,
            total_saved_time=total_saved_time,
            gpu_compute_time_no_cache=self.gpu_compute_no_cache,
            gpu_compute_time_with_cache=gpu_with_cache,
            pcie_total_transfer_bytes=pcie_bytes,
            pcie_total_transfer_time=pcie_time,
            flop_save_rate=flop_save / self.flop_no_cache if self.flop_no_cache > 0 else 0.0,
            total_flop_no_cache=self.flop_no_cache,
            total_flop_with_cache=flop_with_cache,
            hbm_token_hit_rate=hbm_hit_total / total_input if total_input > 0 else 0.0,
            dram_token_hit_rate=dram_hit_total / total_input if total_input > 0 else 0.0,
            per_request_hit_tokens=per_req_hit,
            per_request_dram_hit_tokens=per_req_dram_extra,
            per_request_input_tokens=list(self.req_total_tokens),
            per_request_saved_time=per_req_saved,
            num_binary_vars=num_vars,
            num_constraints=num_constraints,
            solve_time_s=solve_time,
        )
