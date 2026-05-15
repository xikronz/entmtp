"""Runtime tree-selection policy for Hydra speculative decoding.

This module wires together two pieces.

``TopologyBank`` materialises a per-dataset Pareto-frontier of precomputed
Hydra tree topologies (one ``generate_hydra_buffers`` call per tree). The
bank exposes integer indexing and lookup by ``candidate_id`` so a decoding
loop can switch between trees by swapping a dict pointer instead of
rebuilding any tensors per step.

"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from hydra.model.utils import generate_hydra_buffers


@dataclass(frozen=True)
class TopologyMeta:
    """Per-tree metadata loaded from a frontier-throughput JSON."""

    candidate_id: str
    depth: int
    n_nodes: int
    tree: List[List[int]]
    self_rollout_mean_accept: float
    biased_mean_accept: Optional[float]
    output_tokens_per_second: Optional[float]
    seconds_per_decode_step: Optional[float]
    output_tokens_per_decode_step: Optional[float]


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _meta_from_row(row: Mapping[str, Any]) -> TopologyMeta:
    return TopologyMeta(
        candidate_id=str(row["candidate_id"]),
        depth=int(row["depth"]),
        n_nodes=int(row["n_nodes"]),
        tree=[list(p) for p in row["tree"]],
        self_rollout_mean_accept=float(row["self_rollout_mean_accept"]),
        biased_mean_accept=_coerce_float(row.get("biased_mean_accept")),
        output_tokens_per_second=_coerce_float(row.get("output_tokens_per_second")),
        seconds_per_decode_step=_coerce_float(row.get("seconds_per_decode_step")),
        output_tokens_per_decode_step=_coerce_float(row.get("output_tokens_per_decode_step")),
    )


def throughput_pareto_frontier(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Keep only non-dominated points on the throughput hull vs ``n_nodes``.

    Matches ``throughput_frontier`` in ``entmtp/scripts/pareto_throughput.py``:
    sort by ascending ``n_nodes`` (ties break toward higher tok/s), then scan
    and keep a row only if ``output_tokens_per_second`` **strictly** exceeds
    the running best. Larger trees that are slower than a smaller tree are
    dropped even though they appear in ``all_results``.

    ``greedy_*_throughput.json`` files store **every** benchmarked candidate in
    ``all_results``; the dashed hull on the plot is this filtered subset, not
    the raw list.
    """
    usable = [
        r
        for r in rows
        if r.get("output_tokens_per_second") is not None
        and not r.get("is_published_default")
    ]
    pts = sorted(
        usable,
        key=lambda r: (int(r["n_nodes"]), -float(r["output_tokens_per_second"])),
    )
    out: List[Mapping[str, Any]] = []
    best = -float("inf")
    for row in pts:
        score = float(row["output_tokens_per_second"])
        if score > best + 1e-9:
            out.append(row)
            best = score
    return out


class TopologyBank:
    """Holds precomputed ``hydra_buffers`` for every tree in a frontier JSON.

    The frontier JSON is the same artefact produced by the throughput-pareto
    pipeline (e.g. ``entmtp/outputs/pareto_throughput/greedy_humaneval_throughput.json``)

    Important: those JSON files list **every** timed candidate in
    ``all_results``. Many rows are throughput-dominated (larger ``n_nodes`` but
    lower ``output_tokens_per_second`` than some smaller tree). By default
    this bank applies :func:`throughput_pareto_frontier` so the scheduler only
    switches among trees on the same hull used in the throughput plots — not
    the full raw grid (which would let a ladder policy pick huge slow trees).

    Trees are sorted by ``n_nodes`` ascending so that index ``0`` is the
    smallest hull point and index ``-1`` is the largest **on the hull**.
    ``candidate_id`` lookup is also supported.
    """

    def __init__(
        self,
        frontier_results: Union[str, Path, Mapping[str, Any], Sequence[Mapping[str, Any]]],
        device: Union[str, torch.device] = "cuda",
        sort_by: str = "n_nodes",
        filter_ids: Optional[Sequence[str]] = None,
        *,
        only_throughput_pareto: Optional[bool] = None,
    ) -> None:
        rows, payload = self._load_rows(frontier_results)
        if only_throughput_pareto is None:
            only_throughput_pareto = (
                isinstance(payload, Mapping)
                and payload.get("method") == "frontier_throughput"
            )
        if only_throughput_pareto:
            before = len(rows)
            rows = throughput_pareto_frontier(rows)
            if not rows:
                raise ValueError(
                    "TopologyBank: throughput Pareto filter removed all rows; "
                    "check JSON or pass only_throughput_pareto=False to load raw all_results."
                )
            if before > len(rows):
                # One-line hint for operators who diff old vs new behaviour.
                self._pareto_note = (
                    f"filtered all_results {before} -> {len(rows)} "
                    f"(throughput_pareto_frontier)"
                )
            else:
                self._pareto_note = ""
        else:
            self._pareto_note = ""

        if filter_ids is not None:
            keep = set(filter_ids)
            rows = [r for r in rows if str(r["candidate_id"]) in keep]
        if not rows:
            raise ValueError("TopologyBank: no candidate trees after filtering")

        metas = [_meta_from_row(r) for r in rows]
        if sort_by == "n_nodes":
            metas.sort(key=lambda m: (m.n_nodes, m.depth, m.candidate_id))
        elif sort_by == "throughput":
            metas.sort(
                key=lambda m: (
                    m.output_tokens_per_second
                    if m.output_tokens_per_second is not None
                    else float("inf"),
                    m.n_nodes,
                )
            )
        else:
            raise ValueError(f"TopologyBank: unknown sort_by={sort_by!r}")

        self._device = torch.device(device)
        self._meta: List[TopologyMeta] = metas
        self._buffers: List[Dict[str, torch.Tensor]] = [
            generate_hydra_buffers(m.tree, device=self._device) for m in metas
        ]
        self._id_to_index: Dict[str, int] = {
            m.candidate_id: i for i, m in enumerate(metas)
        }

    @property
    def pareto_filter_note(self) -> str:
        """Non-empty when ``only_throughput_pareto`` dropped dominated rows."""
        return getattr(self, "_pareto_note", "")

    @staticmethod
    def _load_rows(
        frontier_results: Union[str, Path, Mapping[str, Any], Sequence[Mapping[str, Any]]],
    ) -> Tuple[List[Mapping[str, Any]], Any]:
        if isinstance(frontier_results, (str, Path)):
            data = json.loads(Path(frontier_results).read_text())
        else:
            data = frontier_results

        if isinstance(data, Mapping):
            if "all_results" not in data:
                raise KeyError(
                    "TopologyBank: frontier dict missing 'all_results' key"
                )
            rows = data["all_results"]
            payload = data
        else:
            rows = data
            payload = None
        rows = [r for r in rows if "tree" in r and "candidate_id" in r]
        return rows, payload

    def __len__(self) -> int:
        return len(self._meta)

    def __iter__(self):
        return iter(self._meta)

    def __getitem__(self, key: Union[int, str]) -> Dict[str, torch.Tensor]:
        return self._buffers[self._index(key)]

    def _index(self, key: Union[int, str]) -> int:
        if isinstance(key, str):
            try:
                return self._id_to_index[key]
            except KeyError as exc:
                raise KeyError(f"TopologyBank: unknown candidate_id {key!r}") from exc
        return int(key)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def metas(self) -> List[TopologyMeta]:
        return list(self._meta)

    @property
    def candidate_ids(self) -> List[str]:
        return [m.candidate_id for m in self._meta]

    def get(self, key: Union[int, str]) -> Dict[str, torch.Tensor]:
        return self[key]

    def get_meta(self, key: Union[int, str]) -> TopologyMeta:
        return self._meta[self._index(key)]

    def tree(self, key: Union[int, str]) -> List[List[int]]:
        return self.get_meta(key).tree

    def depths(self) -> List[int]:
        return [m.depth for m in self._meta]

    def node_counts(self) -> List[int]:
        return [m.n_nodes for m in self._meta]


@dataclass(frozen=True)
class PathValueFeatures:
    """Per-step EAGLE-2 features used by the policy.

    Mirrors the columns ``_add_path_value_features`` in
    ``hydra/log_heads.py`` writes out, so a policy trained on the offline
    notebook features can be applied here without modification.
    """

    base_top1_prob: float
    path_value_greedy: Tuple[float, ...]
    path_value_top2: Tuple[float, ...]
    path_value_gap: Tuple[float, ...]
    best_path_value: float
    best_path_value_depth: int
    path_value_log_sum_exp: float
    path_value_entropy_top4: float


def compute_path_value_features(
    base_logits: torch.Tensor,
    head_logits: torch.Tensor,
) -> PathValueFeatures:
    """Compute path-value features at the final position of ``base_logits``.

    Parameters
    ----------
    base_logits:
        Tensor with shape ``[..., vocab]`` or ``[1, T, vocab]``. The
        next-token logits of the base LM. The final position is used.
    head_logits:
        Tensor with shape ``[num_heads, vocab]`` or
        ``[num_heads, ..., vocab]`` containing each Hydra draft head's logits
        at the same position. ``num_heads`` is the EAGLE depth budget.
    """
    if base_logits.dim() >= 3:
        base_last = base_logits[0, -1]
    elif base_logits.dim() == 2:
        base_last = base_logits[-1]
    else:
        base_last = base_logits

    if head_logits.dim() == 2:
        head_last = head_logits
    elif head_logits.dim() == 3:
        head_last = head_logits[:, -1]
    elif head_logits.dim() == 4:
        head_last = head_logits[:, 0, -1]
    else:
        raise ValueError(
            f"head_logits must have 2-4 dims, got {head_logits.dim()}"
        )

    base_probs = torch.softmax(base_last.float(), dim=-1)
    base_top1_prob = float(base_probs.max().item())

    head_probs = torch.softmax(head_last.float(), dim=-1)
    top2 = torch.topk(head_probs, k=min(2, head_probs.shape[-1]), dim=-1).values
    head_top1 = top2[:, 0].tolist()
    head_top2 = top2[:, 1].tolist() if top2.shape[-1] >= 2 else [0.0] * top2.shape[0]

    num_heads = head_probs.shape[0]
    greedy: List[float] = []
    cum = base_top1_prob
    for d in range(num_heads):
        cum *= float(head_top1[d])
        greedy.append(cum)

    top2_path: List[float] = []
    prefix = base_top1_prob
    for d in range(num_heads):
        if d > 0:
            prefix *= float(head_top1[d - 1])
        top2_path.append(prefix * float(head_top2[d]))

    gap = [g - t2 for g, t2 in zip(greedy, top2_path)]

    if greedy:
        best_val = max(greedy)
        best_depth = greedy.index(best_val) + 1
        if base_top1_prob > best_val:
            best_val = base_top1_prob
            best_depth = 0
    else:
        best_val = base_top1_prob
        best_depth = 0

    path_values = []
    for g, t2 in zip(greedy, top2_path):
        path_values.append(g)
        path_values.append(t2)

    if path_values:
        arr = torch.tensor(path_values, dtype=torch.float64)
        lse = float(torch.logsumexp(torch.log(arr.clamp_min(1e-12)), dim=0).item())
        top4 = torch.topk(arr, k=min(4, arr.numel())).values
        normaliser = top4.sum().clamp_min(1e-12)
        p = top4 / normaliser
        entropy_top4 = float(
            -(p * torch.log(p.clamp_min(1e-12))).sum().item()
        )
    else:
        lse = 0.0
        entropy_top4 = 0.0

    return PathValueFeatures(
        base_top1_prob=base_top1_prob,
        path_value_greedy=tuple(greedy),
        path_value_top2=tuple(top2_path),
        path_value_gap=tuple(gap),
        best_path_value=float(best_val),
        best_path_value_depth=int(best_depth),
        path_value_log_sum_exp=lse,
        path_value_entropy_top4=entropy_top4,
    )


@dataclass(frozen=True)
class TreeAction:
    """Output of :meth:`EagleTreeSelector.select`.

    ``index`` is the integer to look up in the bank; ``candidate_id`` is the
    same tree's frontier name for logging.
    """

    index: int
    candidate_id: str
    features: PathValueFeatures
    score: float


class EagleTreeSelector:
    """Map EAGLE-2 path-value features to a ``TopologyBank`` index.

    Policies:

    ``"threshold_ladder"`` (default) bins a scalar feature (``best_path_value``
    by default; ``path_value_greedy_depth_d`` works too) by ascending
    thresholds, where each bin selects a progressively larger tree. With
    ``n_bins = len(bank)`` and equally spaced thresholds in ``(0, 1)`` this is
    a sensible default that requires no calibration data.

    ``"best_depth"`` selects the tree whose ``depth`` matches the depth at
    which the greedy path value is maximised
    (``features.best_path_value_depth``). Among trees with that depth the
    largest ``n_nodes`` is picked, which matches the throughput-pareto
    intuition: when the head chain has high mass deep, spend the larger node
    budget.

    ``"binary_tau"`` matches the notebook tau experiment: compare
    ``policy_score`` (same scalar as ``score_feature``) to a single ``tau``;
    if ``score > tau`` use the aggressive topology, else the conservative one.
    Both must exist as ``candidate_id`` strings in the bank (defaults:
    ``frontier_0000_n2_d2`` vs ``frontier_0024_n28_d4``).

    Ladder / best-depth modes fall back to the cheapest tree when input
    features are degenerate (e.g. ``num_heads = 0``) and to the largest tree
    when features saturate the top bin.
    """

    def __init__(
        self,
        bank: TopologyBank,
        mode: str = "threshold_ladder",
        thresholds: Optional[Sequence[float]] = None,
        score_feature: str = "best_path_value",
        *,
        tau: float = 0.0,
        binary_conservative_id: str = "frontier_0001_n3_d3",
        binary_aggressive_id: str = "frontier_0024_n28_d4",
    ) -> None:
        if mode not in {"threshold_ladder", "best_depth", "binary_tau"}:
            raise ValueError(
                f"EagleTreeSelector: unknown mode {mode!r}; "
                "expected 'threshold_ladder', 'best_depth', or 'binary_tau'"
            )
        if score_feature not in {
            "best_path_value",
            "path_value_greedy_depth1",
            "path_value_greedy_depth2",
            "path_value_greedy_depth3",
            "path_value_greedy_depth4",
            "base_top1_prob",
        }:
            raise ValueError(
                f"EagleTreeSelector: unknown score_feature {score_feature!r}"
            )

        self.bank = bank
        self.mode = mode
        self.score_feature = score_feature
        self.tau = float(tau)
        self.binary_conservative_id = str(binary_conservative_id)
        self.binary_aggressive_id = str(binary_aggressive_id)

        if mode == "threshold_ladder":
            self._thresholds = self._build_thresholds(thresholds, len(bank))
        else:
            self._thresholds = []

        self._depth_to_index = self._build_depth_table(bank)

        if mode == "binary_tau":
            try:
                self._idx_small = bank._index(self.binary_conservative_id)
                self._idx_big = bank._index(self.binary_aggressive_id)
            except KeyError as exc:
                raise KeyError(
                    "EagleTreeSelector binary_tau: conservative and aggressive "
                    "candidate_id values must exist in the TopologyBank "
                    f"(conservative={self.binary_conservative_id!r}, "
                    f"aggressive={self.binary_aggressive_id!r})"
                ) from exc
        else:
            self._idx_small = -1
            self._idx_big = -1

    @staticmethod
    def _build_thresholds(
        thresholds: Optional[Sequence[float]], n_trees: int
    ) -> List[float]:
        if n_trees <= 1:
            return []
        if thresholds is None:
            step = 1.0 / n_trees
            return [step * (i + 1) for i in range(n_trees - 1)]
        ts = [float(t) for t in thresholds]
        if len(ts) != n_trees - 1:
            raise ValueError(
                f"EagleTreeSelector: thresholds must have len(bank)-1={n_trees - 1} "
                f"entries, got {len(ts)}"
            )
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("EagleTreeSelector: thresholds must be strictly increasing")
        return ts

    @staticmethod
    def _build_depth_table(bank: TopologyBank) -> Dict[int, int]:
        table: Dict[int, int] = {}
        for i, meta in enumerate(bank.metas):
            cur = table.get(meta.depth)
            if cur is None or bank.metas[cur].n_nodes < meta.n_nodes:
                table[meta.depth] = i
        return table

    def _read_score(self, features: PathValueFeatures) -> float:
        if self.score_feature == "best_path_value":
            return features.best_path_value
        if self.score_feature == "base_top1_prob":
            return features.base_top1_prob
        depth = int(self.score_feature.split("depth")[-1])
        if depth - 1 < len(features.path_value_greedy):
            return features.path_value_greedy[depth - 1]
        return features.best_path_value

    def _pick_binary_tau(self, features: PathValueFeatures) -> int:
        score = self._read_score(features)
        if score > self.tau:
            return self._idx_big
        return self._idx_small

    def _pick_threshold_ladder(self, features: PathValueFeatures) -> int:
        score = self._read_score(features)
        for i, t in enumerate(self._thresholds):
            if score < t:
                return i
        return len(self.bank) - 1

    def _pick_best_depth(self, features: PathValueFeatures) -> int:
        d = features.best_path_value_depth
        if d in self._depth_to_index:
            return self._depth_to_index[d]
        available = sorted(self._depth_to_index)
        if not available:
            return 0
        lower = [x for x in available if x <= d]
        if lower:
            return self._depth_to_index[lower[-1]]
        return self._depth_to_index[available[0]]

    def select(
        self,
        base_logits: Optional[torch.Tensor] = None,
        head_logits: Optional[torch.Tensor] = None,
        features: Optional[PathValueFeatures] = None,
    ) -> TreeAction:
        """Return the bank index for this decode step.

        Either pass precomputed ``features`` or both ``base_logits`` and
        ``head_logits`` so they can be computed here. ``head_logits`` must be
        stacked across heads with shape ``[num_heads, ..., vocab]``.
        """
        if features is None:
            if base_logits is None or head_logits is None:
                raise ValueError(
                    "EagleTreeSelector.select: provide either features or both "
                    "base_logits and head_logits"
                )
            features = compute_path_value_features(base_logits, head_logits)

        if self.mode == "threshold_ladder":
            idx = self._pick_threshold_ladder(features)
        elif self.mode == "best_depth":
            idx = self._pick_best_depth(features)
        else:
            idx = self._pick_binary_tau(features)

        return TreeAction(
            index=idx,
            candidate_id=self.bank.get_meta(idx).candidate_id,
            features=features,
            score=self._read_score(features),
        )

    def buffers_for(self, action: TreeAction) -> Dict[str, torch.Tensor]:
        """Convenience: ``bank[action.index]`` for the chosen action."""
        return self.bank[action.index]
