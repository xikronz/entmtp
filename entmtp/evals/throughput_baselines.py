"""Single-benchmark comparison of MTP baselines.

Methods compared (any subset can be skipped via flags):

  * ``vanilla_vicuna``    - base LM autoregressive, no MTP.
  * ``hydra_default``     - Hydra heads + ``mc_sim_7b_63`` (published default tree).
  * ``entmtp_static``     - Hydra heads + the throughput-optimal tree from a
                            frontier JSON (no per-step switching).
  * ``entmtp_scheduled``  - Hydra heads + per-step ``EagleTreeSelector`` over
                            the same frontier (pointer/index switch only).

Reuses the hydra_forward + continuation_ppl helpers from ``throughput.py`` and
the runtime selector from ``entmtp/model/runtime_scheduler.py``.

Timing: pass ``--exclude-prefill`` to match ``entmtp/scripts/pareto_throughput.py``
when ``--exclude-prefill`` is used there (timer starts after ``initialize_hydra`` /
first forward, so prompt prefill is not in ``seconds``).

"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

ROOT = Path("/share/cuvl/cc2864/interpretable/Hydra")
sys.path.insert(0, str(ROOT))

from hydra.model.hydra_choices import mc_sim_7b_63  # noqa: E402
from hydra.model.hydra_model import HydraModel  # noqa: E402
from hydra.model.kv_cache import initialize_past_key_values  # noqa: E402
from hydra.model.utils import (  # noqa: E402
    evaluate_posterior,
    generate_hydra_buffers,
    initialize_hydra,
    reset_hydra_mode,
    tree_decoding,
    update_inference_inputs,
)
from ..scripts.pareto_experiment import load_sharegpt_prompts  # noqa: E402

from entmtp.model.runtime_scheduler import (  # noqa: E402
    EagleTreeSelector,
    PathValueFeatures,
    TopologyBank,
    TopologyMeta,
    compute_path_value_features,
)

def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def continuation_ppl_base(
    model: HydraModel, full_ids: torch.Tensor, prompt_len: int
) -> float:
    """exp(mean cross-entropy) of generated tokens under the base LM."""
    reset_hydra_mode(model)
    with torch.no_grad():
        out = model.base_model(full_ids)
        logits = out.logits[0].float()
    ces: List[float] = []
    for i in range(prompt_len, full_ids.shape[1]):
        ce = F.cross_entropy(
            logits[i - 1 : i], full_ids[0, i : i + 1], reduction="mean"
        )
        ces.append(ce.item())
    if not ces:
        return float("nan")
    return math.exp(sum(ces) / len(ces))


def _head_logits_from_hidden(model: HydraModel, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
    """Return ``[num_heads, vocab]`` head logits at the last position.

    Prefers the cached ``last_proposal_debug`` written by a prior ``proposal``
    call; falls back to a direct head forward for ungrounded MLP / prefix-MLP
    heads (the architecture used by ``ankner/hydra-vicuna-7b-v1.3``). Returns
    ``None`` when neither path is available; callers can then fall back to a
    base-only score feature.
    """
    head = model.hydra_head
    debug = getattr(head, "last_proposal_debug", None)
    if isinstance(debug, dict) and debug.get("head_logits") is not None:
        return debug["head_logits"]

    if model.hydra_head_arch not in {"mlp", "prefix-mlp"}:
        return None
    if getattr(head, "grounded_heads", False):
        return None

    logits = []
    for head_idx in range(model.hydra):
        h = head.hydra_mlp[head_idx](hidden_states)
        logits.append(head.hydra_lm_head[head_idx](h)[0, -1].detach())
    return torch.stack(logits, dim=0)


def _features_base_only(base_logits: torch.Tensor) -> PathValueFeatures:
    """Build a degenerate ``PathValueFeatures`` from base logits only.

    Useful for the first decode step (no head logits yet) and for head
    architectures we cannot probe outside of ``proposal``.
    """
    last = base_logits[0, -1] if base_logits.dim() >= 3 else base_logits[-1]
    probs = torch.softmax(last.float(), dim=-1)
    p = float(probs.max().item())
    return PathValueFeatures(
        base_top1_prob=p,
        path_value_greedy=(p,),
        path_value_top2=(0.0,),
        path_value_gap=(p,),
        best_path_value=p,
        best_path_value_depth=0,
        path_value_log_sum_exp=0.0,
        path_value_entropy_top4=0.0,
    )


def _frontier_for_scheduled_bank(
    frontier_json: str,
    selector_filter_ids: str = "",
    selector_max_nodes: int = 0,
) -> Any:
    """Optionally shrink the scheduled topology bank before buffers are built.

    This is useful for low-latency scheduled runs: materialize only a small
    binary/3-way action set instead of every Pareto-frontier candidate. The
    static baseline still uses the same filtered bank, so if you want the
    static baseline to consider the full frontier, run it separately without
    these scheduled-bank filters.
    """
    keep_ids = {x.strip() for x in selector_filter_ids.split(",") if x.strip()}
    if not keep_ids and selector_max_nodes <= 0:
        return frontier_json

    payload = json.loads(Path(frontier_json).read_text())
    if isinstance(payload, dict):
        rows = list(payload.get("all_results", []))
    else:
        rows = list(payload)

    filtered = []
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        if keep_ids and cid not in keep_ids:
            continue
        if selector_max_nodes > 0 and int(row.get("n_nodes", 10**9)) > selector_max_nodes:
            continue
        filtered.append(row)

    if not filtered:
        raise ValueError(
            "scheduled topology filters removed all candidates; relax "
            "--selector-filter-ids or --selector-max-nodes"
        )

    if isinstance(payload, dict):
        payload = dict(payload)
        payload["all_results"] = filtered
        return payload
    return filtered


@torch.inference_mode()
def run_vanilla(
    model: HydraModel,
    tok,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    max_input_tokens: int,
    exclude_prefill: bool = False,
) -> Dict[str, Any]:
    """Pure autoregressive decoding through the base LM (no speculation)."""
    input_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    if prompt_len > max_input_tokens:
        return {"error": "prompt too long", "prompt_len": prompt_len}

    reset_hydra_mode(model)
    past_kv, _, current_length_data = initialize_past_key_values(
        model.base_model, model.hydra_head_arch
    )
    current_length_data.zero_()

    _cuda_sync()
    t0 = time.perf_counter()

    out = model.base_model(input_ids, past_key_values=past_kv)
    logits = out.logits[0, -1]
    if exclude_prefill:
        _cuda_sync()
        t0 = time.perf_counter()

    generated = input_ids
    new_tokens = 0

    for _ in range(max_new_tokens):
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)
        nid = int(next_id.item())
        new_tokens += 1
        generated = torch.cat([generated, next_id.view(1, 1)], dim=1)
        if nid == tok.eos_token_id:
            break
        out = model.base_model(next_id.view(1, 1), past_key_values=past_kv)
        logits = out.logits[0, -1]

    _cuda_sync()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ppl = continuation_ppl_base(model, generated, prompt_len)
    return {
        "prompt_tokens": prompt_len,
        "gen_tokens": new_tokens,
        "seconds": elapsed,
        "tokens_per_sec": new_tokens / elapsed if elapsed > 0 else 0.0,
        "decode_steps": new_tokens,
        "mean_accept_length": 1.0,
        "continuation_ppl": ppl,
    }


@torch.inference_mode()
def run_static_tree(
    model: HydraModel,
    tok,
    prompt_text: str,
    hydra_choices: Sequence[Sequence[int]],
    max_steps: int,
    temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
    device: torch.device,
    max_input_tokens: int,
    new_token_cap: int,
    cached_buffers: Optional[Dict[str, torch.Tensor]] = None,
    exclude_prefill: bool = False,
) -> Dict[str, Any]:
    """Hydra speculative decoding with a fixed draft-tree topology."""
    input_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    if prompt_len > max_input_tokens:
        return {"error": "prompt too long", "prompt_len": prompt_len}

    if cached_buffers is None:
        buffers = generate_hydra_buffers(hydra_choices, device=device)
    else:
        buffers = cached_buffers

    past_kv, pkv_data, current_length_data = initialize_past_key_values(
        model.base_model, model.hydra_head_arch
    )
    current_length_data.zero_()

    _cuda_sync()
    t0 = time.perf_counter()

    reset_hydra_mode(model)
    hidden_states, logits = initialize_hydra(
        input_ids,
        model,
        buffers["hydra_attn_mask"],
        past_kv,
        buffers["proposal_cross_attn_masks"],
    )
    if exclude_prefill:
        _cuda_sync()
        t0 = time.perf_counter()

    new_token = 0
    n_steps = 0
    accept_lengths: List[int] = []
    for step in range(max_steps):
        to_pass = input_ids if step == 0 else None
        candidates, tree_candidates = model.hydra_head.proposal(
            logits, hidden_states, buffers, past_kv, to_pass
        )
        hidden_states, logits = tree_decoding(
            model,
            tree_candidates,
            past_kv,
            buffers["hydra_position_ids"],
            input_ids,
            buffers["retrieve_indices"],
        )
        best_candidate, accept_length = evaluate_posterior(
            logits,
            candidates,
            temperature,
            posterior_threshold,
            posterior_alpha,
            buffers["max_accepts"],
        )
        accept_lengths.append(int(accept_length))
        input_ids, logits, hidden_states, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            buffers["retrieve_indices"],
            logits,
            hidden_states,
            new_token,
            pkv_data,
            current_length_data,
            model.hydra_head_arch,
        )
        n_steps = step + 1
        if tok.eos_token_id in input_ids[0, prompt_len:].tolist():
            break
        if new_token > new_token_cap:
            break

    _cuda_sync()
    t1 = time.perf_counter()

    gen_len = int(input_ids.shape[1]) - prompt_len
    elapsed = t1 - t0
    ppl = continuation_ppl_base(model, input_ids, prompt_len)
    return {
        "prompt_tokens": prompt_len,
        "gen_tokens": gen_len,
        "seconds": elapsed,
        "tokens_per_sec": gen_len / elapsed if elapsed > 0 else 0.0,
        "decode_steps": n_steps,
        "mean_accept_length": (
            sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
        ),
        "continuation_ppl": ppl,
    }


@torch.inference_mode()
def run_scheduled(
    model: HydraModel,
    tok,
    prompt_text: str,
    selector: EagleTreeSelector,
    max_steps: int,
    temperature: float,
    posterior_threshold: float,
    posterior_alpha: float,
    device: torch.device,
    max_input_tokens: int,
    new_token_cap: int,
    exclude_prefill: bool = False,
    *,
    fast_policy: bool = True,
) -> Dict[str, Any]:
    """Hydra speculative decoding with per-step tree switching.

    A single ``TopologyBank`` is precompiled and lives inside ``selector``.
    Each decode step is just a pointer swap: select an index, set the model's
    ``hydra_mask`` to that tree's mask, and call the existing proposal /
    tree_decoding / verify path with that tree's buffers.
    """
    bank = selector.bank
    input_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    if prompt_len > max_input_tokens:
        return {"error": "prompt too long", "prompt_len": prompt_len}

    past_kv, pkv_data, current_length_data = initialize_past_key_values(
        model.base_model, model.hydra_head_arch
    )
    current_length_data.zero_()

    largest = bank[len(bank) - 1]

    _cuda_sync()
    t0 = time.perf_counter()

    reset_hydra_mode(model)
    hidden_states, logits = initialize_hydra(
        input_ids,
        model,
        largest["hydra_attn_mask"],
        past_kv,
        largest["proposal_cross_attn_masks"],
    )
    if exclude_prefill:
        _cuda_sync()
        t0 = time.perf_counter()

    selections: List[int] = []
    accept_lengths: List[int] = []
    policy_scores: List[float] = []
    new_token = 0
    n_steps = 0

    # ``runtime_scheduler_latency_optimized.py`` exposes select_fast(),
    # policy_period, hysteresis, and buffers_for_index(). Use that path when
    # available; otherwise keep the original full-feature fallback.
    has_fast_selector = fast_policy and hasattr(selector, "select_fast")
    policy_period = max(1, int(getattr(selector, "policy_period", 1)))

    for step in range(max_steps):
        if has_fast_selector:
            # Avoid probing Hydra head logits on steps where the scheduler will
            # reuse the previous tree index. select_fast(step=...) skips all
            # score computation when step % policy_period != 0.
            needs_policy_update = step == 0 or (step % policy_period == 0)
            head_logits = _head_logits_from_hidden(model, hidden_states) if needs_policy_update else None
            if needs_policy_update and head_logits is None and selector.mode == "best_depth":
                # best_depth needs the full feature object; if head logits are
                # unavailable, fall back to a conservative base-only feature.
                action = selector.select(features=_features_base_only(logits))
                selected_index = action.index
                if not _is_nan(action.score):
                    policy_scores.append(float(action.score))
            else:
                selected_index = int(
                    selector.select_fast(
                        logits,
                        head_logits,
                        step=step,
                        return_action=False,
                    )
                )
                score = getattr(selector, "_last_score", float("nan"))
                if needs_policy_update and not _is_nan(score):
                    policy_scores.append(float(score))
            buffers = (
                selector.buffers_for_index(selected_index)
                if hasattr(selector, "buffers_for_index")
                else bank[selected_index]
            )
        else:
            head_logits = _head_logits_from_hidden(model, hidden_states)
            if head_logits is None:
                features = _features_base_only(logits)
            else:
                features = compute_path_value_features(logits, head_logits)
            action = selector.select(features=features)
            selected_index = action.index
            if not _is_nan(action.score):
                policy_scores.append(float(action.score))
            buffers = bank[selected_index]

        selections.append(selected_index)

        model.base_model.model.hydra_mask = buffers["hydra_attn_mask"]
        if model.hydra_head_arch == "cross-attn":
            model.hydra_head.proposal_hydra_masks = buffers[
                "proposal_cross_attn_masks"
            ]

        to_pass = input_ids if step == 0 else None
        candidates, tree_candidates = model.hydra_head.proposal(
            logits, hidden_states, buffers, past_kv, to_pass
        )
        hidden_states, logits = tree_decoding(
            model,
            tree_candidates,
            past_kv,
            buffers["hydra_position_ids"],
            input_ids,
            buffers["retrieve_indices"],
        )
        best_candidate, accept_length = evaluate_posterior(
            logits,
            candidates,
            temperature,
            posterior_threshold,
            posterior_alpha,
            buffers["max_accepts"],
        )
        accept_lengths.append(int(accept_length))
        input_ids, logits, hidden_states, new_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            buffers["retrieve_indices"],
            logits,
            hidden_states,
            new_token,
            pkv_data,
            current_length_data,
            model.hydra_head_arch,
        )
        n_steps = step + 1
        if tok.eos_token_id in input_ids[0, prompt_len:].tolist():
            break
        if new_token > new_token_cap:
            break

    _cuda_sync()
    t1 = time.perf_counter()

    gen_len = int(input_ids.shape[1]) - prompt_len
    elapsed = t1 - t0
    ppl = continuation_ppl_base(model, input_ids, prompt_len)

    histogram: Dict[str, int] = {}
    for idx in selections:
        cid = bank.get_meta(idx).candidate_id
        histogram[cid] = histogram.get(cid, 0) + 1

    return {
        "prompt_tokens": prompt_len,
        "gen_tokens": gen_len,
        "seconds": elapsed,
        "tokens_per_sec": gen_len / elapsed if elapsed > 0 else 0.0,
        "decode_steps": n_steps,
        "mean_accept_length": (
            sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
        ),
        "continuation_ppl": ppl,
        "tree_selection_histogram": histogram,
        "fast_policy": bool(has_fast_selector),
        "policy_period": int(policy_period),
        "mean_policy_score": (
            sum(policy_scores) / len(policy_scores) if policy_scores else float("nan")
        ),
        "mean_selected_n_nodes": (
            sum(bank.get_meta(i).n_nodes for i in selections) / len(selections)
            if selections else 0.0
        ),
    }


def load_hydra_like(checkpoint: str, base_model: str) -> HydraModel:
    print(f"Loading {checkpoint} (base={base_model}) ...")
    model = HydraModel.from_pretrained(
        checkpoint,
        base_model=base_model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return model


def free_model(model: Optional[Any]) -> None:
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _skipped_baseline_agg(reason: str) -> Dict[str, Any]:
    """Placeholder summary row when a baseline was not run (still appears in JSON)."""
    return {
        "n": 0,
        "n_error": 0,
        "skipped": True,
        "skip_reason": reason,
        "mean_tokens_per_sec": float("nan"),
        "mean_gen_tokens": float("nan"),
        "mean_seconds": float("nan"),
        "mean_decode_steps": float("nan"),
        "mean_accept_length": float("nan"),
        "mean_continuation_ppl": float("nan"),
    }


def pick_throughput_optimal(bank: TopologyBank) -> TopologyMeta:
    """Choose the frontier tree with the highest measured tok/s."""
    metas = bank.metas
    scored = [
        (m.output_tokens_per_second if m.output_tokens_per_second is not None else -1.0, m)
        for m in metas
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n": 0, "n_error": len(rows)}

    def mean(key: str) -> float:
        xs = [r[key] for r in ok if r.get(key) is not None and not _is_nan(r[key])]
        return sum(xs) / len(xs) if xs else float("nan")

    agg = {
        "n": len(ok),
        "n_error": len(rows) - len(ok),
        "mean_tokens_per_sec": mean("tokens_per_sec"),
        "mean_gen_tokens": mean("gen_tokens"),
        "mean_seconds": mean("seconds"),
        "mean_decode_steps": mean("decode_steps"),
        "mean_accept_length": mean("mean_accept_length"),
        "mean_continuation_ppl": mean("continuation_ppl"),
        "mean_selected_n_nodes": mean("mean_selected_n_nodes"),
        "mean_policy_score": mean("mean_policy_score"),
    }
    sel_histograms = [
        r["tree_selection_histogram"] for r in ok if "tree_selection_histogram" in r
    ]
    if sel_histograms:
        combined: Dict[str, int] = {}
        for h in sel_histograms:
            for k, v in h.items():
                combined[k] = combined.get(k, 0) + v
        total = sum(combined.values())
        agg["tree_selection_share"] = {
            k: v / total for k, v in sorted(combined.items())
        }
    return agg


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def print_summary(results: Dict[str, Dict[str, Any]]) -> None:
    print("\n=== Aggregated results ===")
    header = (
        f"{'method':<22s}  {'n':>3s}  {'tok/s':>9s}  {'gen_tok':>7s}  "
        f"{'steps':>5s}  {'accept':>6s}  {'ppl':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name, agg in results.items():
        if agg.get("skipped"):
            print(f"{name:<22s}  SKIPPED  ({agg.get('skip_reason', '')})")
            continue
        if agg.get("n", 0) == 0:
            print(f"{name:<22s}  (no successful prompts)")
            continue
        print(
            f"{name:<22s}  {agg['n']:>3d}  "
            f"{agg['mean_tokens_per_sec']:>9.2f}  "
            f"{agg['mean_gen_tokens']:>7.1f}  "
            f"{agg['mean_decode_steps']:>5.1f}  "
            f"{agg['mean_accept_length']:>6.2f}  "
            f"{agg['mean_continuation_ppl']:>7.2f}"
        )

    for name, agg in results.items():
        share = agg.get("tree_selection_share")
        if share:
            print(f"\n{name} tree-selection share:")
            for cid, frac in share.items():
                print(f"  {cid:<28s}  {frac:6.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, help="Benchmark JSON/JSONL.")
    p.add_argument(
        "--frontier-json",
        required=True,
        help="frontier-throughput JSON (e.g. greedy_humaneval_throughput.json).",
    )
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--pool-size", type=int, default=2000)
    p.add_argument("--sample-seed", type=int, default=123)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-input-tokens", type=int, default=1400)
    p.add_argument("--new-token-cap", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--posterior-threshold", type=float, default=0.09)
    p.add_argument("--posterior-alpha", type=float, default=0.3)
    p.add_argument(
        "--exclude-prefill",
        action="store_true",
        help=(
            "Start the perf timer after initialize_hydra (Hydra) or after the "
            "first base forward (vanilla), matching entmtp/scripts/pareto_throughput.py "
            "with --exclude-prefill and frontier JSON include_prefill=false."
        ),
    )

    p.add_argument("--hydra-checkpoint", default="ankner/hydra-vicuna-7b-v1.3")
    p.add_argument("--base-model", default="lmsys/vicuna-7b-v1.3")

    p.add_argument(
        "--selector-mode",
        default="threshold_ladder",
        choices=("threshold_ladder", "best_depth", "binary_tau"),
    )
    p.add_argument(
        "--policy-tau",
        type=float,
        default=0.0,
        help="For binary_tau: use aggressive tree iff score_feature > this value.",
    )
    p.add_argument(
        "--binary-conservative-id",
        default="frontier_0000_n2_d2",
        help="For binary_tau: candidate_id when score <= tau.",
    )
    p.add_argument(
        "--binary-aggressive-id",
        default="frontier_0024_n28_d4",
        help="For binary_tau: candidate_id when score > tau.",
    )
    p.add_argument(
        "--score-feature",
        default="best_path_value",
        choices=(
            "best_path_value",
            "base_top1_prob",
            "path_value_greedy_depth1",
            "path_value_greedy_depth2",
            "path_value_greedy_depth3",
            "path_value_greedy_depth4",
        ),
    )
    p.add_argument(
        "--selector-thresholds",
        default="",
        help="Comma-separated thresholds; default equally spaced in (0, 1).",
    )

    p.add_argument(
        "--policy-every",
        type=int,
        default=1,
        help="Run the topology policy every N decode steps and reuse the prior tree between updates.",
    )
    p.add_argument(
        "--binary-tau-on",
        type=float,
        default=None,
        help="For binary_tau hysteresis: switch conservative -> aggressive at this score.",
    )
    p.add_argument(
        "--binary-tau-off",
        type=float,
        default=None,
        help="For binary_tau hysteresis: switch aggressive -> conservative at this score.",
    )
    p.add_argument(
        "--disable-fast-policy",
        action="store_true",
        help="Use the old full PathValueFeatures path instead of selector.select_fast().",
    )
    p.add_argument(
        "--selector-filter-ids",
        default="",
        help="Comma-separated candidate_ids to materialize for scheduled/static EntMTP runs.",
    )
    p.add_argument(
        "--selector-max-nodes",
        type=int,
        default=0,
        help="If >0, drop topology candidates with n_nodes above this cap before building buffers.",
    )

    p.add_argument("--skip-vanilla", action="store_true")
    p.add_argument("--skip-hydra", action="store_true")
    p.add_argument("--skip-entmtp-static", action="store_true")
    p.add_argument("--skip-entmtp-scheduled", action="store_true")

    p.add_argument("--out-json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; run on a GPU node.")

    pool = load_sharegpt_prompts(
        path=args.data_path, n=max(args.pool_size, args.n_prompts), seed=123
    )
    if len(pool) < args.n_prompts:
        raise SystemExit(
            f"Only got {len(pool)} prompts; increase --pool-size or lower --n-prompts"
        )
    rng = random.Random(args.sample_seed)
    rng.shuffle(pool)
    prompts = [p["prompt_text"] for p in pool[: args.n_prompts]]
    print(
        f"data_path={args.data_path}  n_prompts={len(prompts)}  "
        f"max_new_tokens={args.max_new_tokens}  "
        f"tau={args.temperature} eps={args.posterior_threshold} alpha={args.posterior_alpha}  "
        f"exclude_prefill={args.exclude_prefill}"
    )

    results: Dict[str, Dict[str, Any]] = {}
    raw_rows: Dict[str, List[Dict[str, Any]]] = {}

    hydra_runs = (
        not args.skip_vanilla
        or not args.skip_hydra
        or not args.skip_entmtp_static
        or not args.skip_entmtp_scheduled
    )
    if hydra_runs:
        try:
            hydra_model = load_hydra_like(args.hydra_checkpoint, args.base_model)
            tok = hydra_model.get_tokenizer()
            device = hydra_model.base_model.device

            frontier_for_bank = _frontier_for_scheduled_bank(
                args.frontier_json,
                selector_filter_ids=args.selector_filter_ids,
                selector_max_nodes=args.selector_max_nodes,
            )
            bank = TopologyBank(frontier_for_bank, device=device)
            static_meta = pick_throughput_optimal(bank)
            print(
                f"frontier trees={len(bank)}  "
                f"pareto_note={bank.pareto_filter_note!r}  "
                f"static throughput-optimal={static_meta.candidate_id} "
                f"(n={static_meta.n_nodes}, d={static_meta.depth}, "
                f"tok/s={static_meta.output_tokens_per_second})"
            )

            thresholds: Optional[List[float]] = None
            if args.selector_thresholds.strip():
                thresholds = [float(x) for x in args.selector_thresholds.split(",")]
            selector = EagleTreeSelector(
                bank,
                mode=args.selector_mode,
                thresholds=thresholds,
                score_feature=args.score_feature,
                tau=args.policy_tau,
                binary_conservative_id=args.binary_conservative_id,
                binary_aggressive_id=args.binary_aggressive_id,
                tau_on=args.binary_tau_on,
                tau_off=args.binary_tau_off,
                policy_period=args.policy_every,
            )

            def bench(name: str, fn) -> None:
                rows = []
                print(f"\n>> {name}")
                for i, text in enumerate(prompts):
                    try:
                        row = fn(text)
                    except Exception as exc:  # noqa: BLE001
                        row = {"error": str(exc)}
                    rows.append(row)
                    if "error" in row:
                        print(f"  [{i + 1}/{len(prompts)}] ERROR {row['error']}")
                    else:
                        print(
                            f"  [{i + 1}/{len(prompts)}] "
                            f"gen={row['gen_tokens']:3d} "
                            f"tok/s={row['tokens_per_sec']:7.2f} "
                            f"steps={row['decode_steps']:3d} "
                            f"acc={row['mean_accept_length']:.2f} "
                            f"ppl={row['continuation_ppl']:.2f}"
                        )
                raw_rows[name] = rows
                results[name] = aggregate(rows)

            if not args.skip_vanilla:
                bench(
                    "vanilla_vicuna",
                    lambda text: run_vanilla(
                        hydra_model,
                        tok,
                        text,
                        args.max_new_tokens,
                        args.temperature,
                        device,
                        args.max_input_tokens,
                        exclude_prefill=args.exclude_prefill,
                    ),
                )
            if not args.skip_hydra:
                default_buffers = generate_hydra_buffers(mc_sim_7b_63, device=device)
                bench(
                    "hydra_default",
                    lambda text: run_static_tree(
                        hydra_model,
                        tok,
                        text,
                        mc_sim_7b_63,
                        args.max_new_tokens,
                        args.temperature,
                        args.posterior_threshold,
                        args.posterior_alpha,
                        device,
                        args.max_input_tokens,
                        args.new_token_cap,
                        cached_buffers=default_buffers,
                        exclude_prefill=args.exclude_prefill,
                    ),
                )
            if not args.skip_entmtp_static:
                static_buffers = bank[static_meta.candidate_id]
                bench(
                    f"entmtp_static[{static_meta.candidate_id}]",
                    lambda text: run_static_tree(
                        hydra_model,
                        tok,
                        text,
                        static_meta.tree,
                        args.max_new_tokens,
                        args.temperature,
                        args.posterior_threshold,
                        args.posterior_alpha,
                        device,
                        args.max_input_tokens,
                        args.new_token_cap,
                        cached_buffers=static_buffers,
                        exclude_prefill=args.exclude_prefill,
                    ),
                )
            if not args.skip_entmtp_scheduled:
                bench(
                    "entmtp_scheduled",
                    lambda text: run_scheduled(
                        hydra_model,
                        tok,
                        text,
                        selector,
                        args.max_new_tokens,
                        args.temperature,
                        args.posterior_threshold,
                        args.posterior_alpha,
                        device,
                        args.max_input_tokens,
                        args.new_token_cap,
                        exclude_prefill=args.exclude_prefill,
                        fast_policy=not args.disable_fast_policy,
                    ),
                )

            free_model(hydra_model)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] Hydra / frontier benchmark block failed; continuing ({exc})."
            )

    print_summary(results)

    if args.out_json:
        payload = {
            "args": vars(args),
            "n_prompts": len(prompts),
            "summary": results,
            "rows": raw_rows,
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()