"""PathSnap Sequential Route Reconstruction — Production HMM & Viterbi Decoder (Member 3).

Combines Member 2's calibrated classifier probabilities (emissions) with road-network
connectivity and shortest-path routing (transitions) inside a log-space Viterbi decoder.

Key features:
  - Hidden states: candidate road segments (`arc_id`) per GPS observation.
  - Emissions: `emission_weight * log(clip(p_clf, eps, 1 - eps))` from the tuned classifier.
  - Transitions: directed road-network connectivity (same-arc stay, direct edge
    `to_node(a_i) == from_node(a_j)`, or bounded local Dijkstra multi-hop path) scored
    with Newson-Krumm exponential detour penalty `-|d_road - d_gps| / beta`.
  - Edge-case handling:
      * Disconnected / unreachable transitions receive a finite `disconnect_penalty`
        (never -inf) to prevent Viterbi deadlocks.
      * Large GPS time gaps (`dt > gap_threshold_s`) trigger a soft transition reset
        so dropouts do not corrupt downstream route decoding.
      * Missing candidate observations are gracefully skipped/carried forward.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class HMMConfig:
    """Tunable HMM/Viterbi decoder parameters (tuned strictly on permitted dev data)."""

    emission_weight: float = 1.0
    beta: float = 15.0
    self_transition_bonus: float = 0.5
    direct_edge_bonus: float = 0.2
    disconnect_penalty: float = -12.0
    gap_threshold_s: float = 5.0
    max_search_multiplier: float = 2.5
    min_search_radius_m: float = 120.0
    max_search_radius_m: float = 600.0
    prob_eps: float = 1e-6

    def to_dict(self) -> Dict[str, float]:
        return {
            "emission_weight": float(self.emission_weight),
            "beta": float(self.beta),
            "self_transition_bonus": float(self.self_transition_bonus),
            "direct_edge_bonus": float(self.direct_edge_bonus),
            "disconnect_penalty": float(self.disconnect_penalty),
            "gap_threshold_s": float(self.gap_threshold_s),
            "max_search_multiplier": float(self.max_search_multiplier),
            "min_search_radius_m": float(self.min_search_radius_m),
            "max_search_radius_m": float(self.max_search_radius_m),
            "prob_eps": float(self.prob_eps),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "HMMConfig":
        valid = {k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class RoadNetworkGraph:
    """Lightweight directed weighted road graph for fast bounded Dijkstra queries."""

    def __init__(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        from_nodes: np.ndarray,
        to_nodes: np.ndarray,
    ) -> None:
        self.node_x = np.asarray(node_x, dtype=np.float64)
        self.node_y = np.asarray(node_y, dtype=np.float64)
        self.from_nodes = np.asarray(from_nodes, dtype=np.int64)
        self.to_nodes = np.asarray(to_nodes, dtype=np.int64)

        n_nodes = len(self.node_x)
        valid = (
            (self.from_nodes >= 0)
            & (self.to_nodes >= 0)
            & (self.from_nodes < n_nodes)
            & (self.to_nodes < n_nodes)
        )
        dx = np.zeros(len(self.from_nodes), dtype=np.float64)
        dy = np.zeros(len(self.from_nodes), dtype=np.float64)
        dx[valid] = self.node_x[self.to_nodes[valid]] - self.node_x[self.from_nodes[valid]]
        dy[valid] = self.node_y[self.to_nodes[valid]] - self.node_y[self.from_nodes[valid]]
        self.arc_lengths = np.hypot(dx, dy)

        # Build compact adjacency list: u -> list of (v, edge_len_m)
        adj: Dict[int, List[Tuple[int, float]]] = {}
        for u, v, length in zip(
            self.from_nodes[valid], self.to_nodes[valid], self.arc_lengths[valid]
        ):
            u_i, v_i = int(u), int(v)
            if u_i not in adj:
                adj[u_i] = [(v_i, float(length))]
            else:
                adj[u_i].append((v_i, float(length)))
        self.adj = adj

    def bounded_dijkstra(
        self,
        source_node: int,
        targets: Sequence[int],
        cutoff_m: float,
        max_visited: int = 500,
    ) -> Dict[int, float]:
        """Compute shortest directed road distance from source_node to targets within cutoff_m."""
        target_set = set(int(t) for t in targets)
        if not target_set:
            return {}

        dist: Dict[int, float] = {int(source_node): 0.0}
        found: Dict[int, float] = {}
        if int(source_node) in target_set:
            found[int(source_node)] = 0.0
            if len(found) == len(target_set):
                return found

        heap: List[Tuple[float, int]] = [(0.0, int(source_node))]
        adj = self.adj
        visited = 0

        while heap and visited < max_visited:
            d, u = heapq.heappop(heap)
            if d > cutoff_m:
                break
            if d > dist.get(u, math.inf):
                continue
            visited += 1
            if u in target_set and u not in found:
                found[u] = d
                if len(found) == len(target_set):
                    break
            neighbors = adj.get(u)
            if not neighbors:
                continue
            for v, w in neighbors:
                nd = d + w
                if nd <= cutoff_m and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        return found


class HMMRouteDecoder:
    """Log-space Hidden Markov Model + Viterbi route reconstructor for PathSnap."""

    def __init__(self, config: Optional[HMMConfig] = None) -> None:
        self.config = config or HMMConfig()

    def _emission_log_probs(self, probs: np.ndarray) -> np.ndarray:
        eps = self.config.prob_eps
        clipped = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
        return self.config.emission_weight * np.log(clipped)

    def _transition_log_matrix(
        self,
        prev_arcs: np.ndarray,
        prev_from: np.ndarray,
        prev_to: np.ndarray,
        curr_arcs: np.ndarray,
        curr_from: np.ndarray,
        curr_to: np.ndarray,
        gps_step_m: float,
        dt_s: float,
        graph: Optional[RoadNetworkGraph],
    ) -> np.ndarray:
        """Compute log transition matrix of shape (N_prev, N_curr)."""
        n_prev = len(prev_arcs)
        n_curr = len(curr_arcs)
        cfg = self.config

        # Soft reset across large GPS dropouts / irregular time gaps
        if not math.isfinite(dt_s) or dt_s > cfg.gap_threshold_s or dt_s < 0:
            return np.zeros((n_prev, n_curr), dtype=np.float64)

        step_m = float(gps_step_m) if math.isfinite(gps_step_m) and gps_step_m >= 0 else 0.0
        log_T = np.full((n_prev, n_curr), cfg.disconnect_penalty, dtype=np.float64)

        # Fast vectorized checks: same arc or direct edge to_node(i) == from_node(j)
        same_arc = prev_arcs[:, None] == curr_arcs[None, :]
        direct_edge = (prev_to[:, None] >= 0) & (prev_to[:, None] == curr_from[None, :])

        # Multi-hop bounded Dijkstra ONLY if graph is available and needed
        cutoff_m = min(
            cfg.max_search_radius_m,
            max(cfg.min_search_radius_m, cfg.max_search_multiplier * step_m + 30.0),
        )

        if graph is not None:
            unique_targets = np.unique(curr_from[curr_from >= 0]).tolist()
            dijkstra_cache: Dict[int, Dict[int, float]] = {}

            for i in range(n_prev):
                u_end = int(prev_to[i])
                if u_end < 0:
                    continue
                if u_end not in dijkstra_cache:
                    # Only run Dijkstra if there are non-same, non-direct candidates
                    need_targets = [
                        int(curr_from[j])
                        for j in range(n_curr)
                        if not same_arc[i, j] and not direct_edge[i, j] and curr_from[j] >= 0
                    ]
                    if need_targets:
                        dijkstra_cache[u_end] = graph.bounded_dijkstra(
                            u_end, need_targets, cutoff_m=cutoff_m
                        )
                    else:
                        dijkstra_cache[u_end] = {}

                dists = dijkstra_cache[u_end]
                if not dists:
                    continue
                for j in range(n_curr):
                    if same_arc[i, j] or direct_edge[i, j]:
                        continue
                    cf = int(curr_from[j])
                    if cf in dists:
                        # Approximate road distance between midpoints/endpoints:
                        # shortest path from to_node(i) to from_node(j)
                        road_gap = dists[cf]
                        detour = abs(road_gap - step_m)
                        score = -detour / cfg.beta
                        if score > log_T[i, j]:
                            log_T[i, j] = score

        # Apply direct edge transitions: road distance between endpoints is 0
        if np.any(direct_edge):
            direct_score = cfg.direct_edge_bonus - (abs(step_m) * 0.25) / cfg.beta
            log_T = np.where(direct_edge, np.maximum(log_T, direct_score), log_T)

        # Apply same-arc stay transitions (highest topological continuity)
        if np.any(same_arc):
            log_T = np.where(same_arc, cfg.self_transition_bonus, log_T)

        return log_T

    def decode_trip(
        self,
        trip_candidates: pd.DataFrame,
        gps_x: Optional[np.ndarray] = None,
        gps_y: Optional[np.ndarray] = None,
        timestamps_s: Optional[np.ndarray] = None,
        graph: Optional[RoadNetworkGraph] = None,
    ) -> pd.DataFrame:
        """Run log-space Viterbi decoding for a single trip's candidate probability table.

        Parameters
        ----------
        trip_candidates : DataFrame
            Must contain columns: `obs_idx`, `arc_id`, `from_node`, `to_node`, `prob`.
            Optional: `label`.
        gps_x, gps_y, timestamps_s : optional 1D arrays indexed by `obs_idx`.
        graph : optional RoadNetworkGraph for bounded multi-hop shortest-path transitions.

        Returns
        -------
        DataFrame with one row per decoded observation:
            `obs_idx`, `arc_id`, `from_node`, `to_node`, `prob`, (`label` if present).
        """
        if trip_candidates.empty:
            return trip_candidates.copy()

        # Group candidates cleanly by sorted obs_idx
        grouped_obs: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
        has_label = "label" in trip_candidates.columns

        for obs_idx, grp in trip_candidates.groupby("obs_idx", sort=True):
            arcs = grp["arc_id"].to_numpy(dtype=np.int64)
            f_nodes = grp["from_node"].to_numpy(dtype=np.int64)
            t_nodes = grp["to_node"].to_numpy(dtype=np.int64)
            probs = grp["prob"].to_numpy(dtype=np.float64)
            labels = grp["label"].to_numpy(dtype=np.int8) if has_label else None
            grouped_obs.append((int(obs_idx), arcs, f_nodes, t_nodes, probs, labels))

        T = len(grouped_obs)
        if T == 1:
            obs_idx, arcs, f_nodes, t_nodes, probs, labels = grouped_obs[0]
            best = int(np.argmax(probs))
            row = {
                "obs_idx": obs_idx,
                "arc_id": int(arcs[best]),
                "from_node": int(f_nodes[best]),
                "to_node": int(t_nodes[best]),
                "prob": float(probs[best]),
            }
            if has_label and labels is not None:
                row["label"] = int(labels[best])
            return pd.DataFrame([row])

        # Forward Viterbi recursion in log-space
        dp_prev = self._emission_log_probs(grouped_obs[0][4])
        backpointers: List[np.ndarray] = []

        for t in range(1, T):
            prev_obs_idx, prev_arcs, prev_from, prev_to, _, _ = grouped_obs[t - 1]
            curr_obs_idx, curr_arcs, curr_from, curr_to, curr_probs, _ = grouped_obs[t]

            # Compute Euclidean GPS step and delta-t if track arrays provided
            if (
                gps_x is not None
                and gps_y is not None
                and prev_obs_idx < len(gps_x)
                and curr_obs_idx < len(gps_x)
            ):
                gps_step_m = float(
                    math.hypot(
                        gps_x[curr_obs_idx] - gps_x[prev_obs_idx],
                        gps_y[curr_obs_idx] - gps_y[prev_obs_idx],
                    )
                )
            else:
                gps_step_m = 10.0

            if (
                timestamps_s is not None
                and prev_obs_idx < len(timestamps_s)
                and curr_obs_idx < len(timestamps_s)
            ):
                dt_s = float(timestamps_s[curr_obs_idx] - timestamps_s[prev_obs_idx])
            else:
                dt_s = float(curr_obs_idx - prev_obs_idx)

            log_T = self._transition_log_matrix(
                prev_arcs=prev_arcs,
                prev_from=prev_from,
                prev_to=prev_to,
                curr_arcs=curr_arcs,
                curr_from=curr_from,
                curr_to=curr_to,
                gps_step_m=gps_step_m,
                dt_s=dt_s,
                graph=graph,
            )

            # scores[i, j] = dp_prev[i] + log_T[i, j]
            scores = dp_prev[:, None] + log_T
            best_prev = np.argmax(scores, axis=0)
            best_scores = scores[best_prev, np.arange(len(curr_arcs))]
            dp_curr = best_scores + self._emission_log_probs(curr_probs)

            # Stabilize log-scores to prevent drift on long trips (>2000 obs)
            max_val = float(np.max(dp_curr))
            if math.isfinite(max_val):
                dp_curr = dp_curr - max_val

            backpointers.append(best_prev.astype(np.int32))
            dp_prev = dp_curr

        # Backtrack optimal state sequence
        chosen_idx = np.zeros(T, dtype=np.int32)
        chosen_idx[T - 1] = int(np.argmax(dp_prev))
        for t in range(T - 1, 0, -1):
            chosen_idx[t - 1] = backpointers[t - 1][chosen_idx[t]]

        rows = []
        for t in range(T):
            obs_idx, arcs, f_nodes, t_nodes, probs, labels = grouped_obs[t]
            k = int(chosen_idx[t])
            row = {
                "obs_idx": obs_idx,
                "arc_id": int(arcs[k]),
                "from_node": int(f_nodes[k]),
                "to_node": int(t_nodes[k]),
                "prob": float(probs[k]),
            }
            if has_label and labels is not None:
                row["label"] = int(labels[k])
            rows.append(row)

        return pd.DataFrame(rows)
