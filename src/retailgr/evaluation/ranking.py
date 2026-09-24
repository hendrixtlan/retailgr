"""Evaluating the ranker.

Two questions, and they are not the same:

1. **Is each head calibrated?** Measured as ROC AUC per head on held-out
   positions, teacher-forced on the observed actions. A purchase head with an
   AUC near 0.5 is not ranking anything, whatever the blended score does.
2. **Does re-ranking beat not re-ranking?** Retrieval already produces an
   ordering. The ranker is only worth its latency if reordering the same
   candidate set improves the metric. That comparison — same candidates, same
   targets, two orderings — is the only one that answers whether the second
   stage earns its place.

Question 2 is the one teams skip, and it is the one that decides whether to
ship the ranker.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from retailgr.actions import ACTION_TO_ID
from retailgr.evaluation.calibration import (
    DEFAULT_BINS as DEFAULT_CALIBRATION_BINS,
)
from retailgr.evaluation.calibration import (
    HeadCalibrators,
    calibration_report,
    fit_calibrator,
)
from retailgr.evaluation.metrics import ndcg_at_k, recall_at_k
from retailgr.evaluation.stats import PairedComparison, paired_comparison
from retailgr.io.loaders import SequenceSplit
from retailgr.models.ranker import (
    HEAD_POSITIVE_ACTIONS,
    HEADS,
    HSTURanker,
)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the rank-sum identity, with ties averaged.

    Written out rather than imported so the package has no scikit-learn
    dependency for one function.
    """
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = labels > 0
    n_pos = int(positives.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        # AUC is undefined with one class; say so rather than returning 0.5,
        # which looks like a measured result.
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    # Average the ranks inside each group of equal scores.
    start = 0
    for index in range(1, sorted_scores.size + 1):
        if index == sorted_scores.size or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                tied = order[start:index]
                ranks[tied] = ranks[tied].mean()
            start = index

    rank_sum = ranks[positives].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_heads(
    ranker: HSTURanker, split: SequenceSplit, batch_size: int = 64
) -> dict[str, Any]:
    """Per-head AUC on the observed actions of a held-out split."""
    import torch

    collected: dict[str, list[np.ndarray]] = {name: [] for name in HEADS}
    labels_collected: dict[str, list[np.ndarray]] = {name: [] for name in HEADS}

    usable = [
        index
        for index, history in enumerate(split.inputs)
        if ((history > 0) & (history < ranker.vocab_size)).sum() >= 2
    ]

    ranker.net.eval()
    with torch.no_grad():
        for start in range(0, len(usable), batch_size):
            indices = usable[start : start + batch_size]
            batch = ranker._padded_history(split, indices)
            tokens = batch["tokens"].to(ranker.device)
            actions = batch["actions"].to(ranker.device)
            times = batch["timestamps"].to(ranker.device)
            returned = batch["returned"].to(ranker.device)

            logits, _ = ranker.net(tokens, actions, times)
            label_map = ranker._labels_for(actions, returned)
            for head in HEADS:
                label, mask = label_map[head]
                head_logits = logits[head][:, 0::2]
                keep = mask > 0
                if not bool(keep.any()):
                    continue
                collected[head].append(head_logits[keep].float().cpu().numpy())
                labels_collected[head].append(label[keep].float().cpu().numpy())

    out: dict[str, Any] = {}
    for head in HEADS:
        if not collected[head]:
            out[head] = {"auc": float("nan"), "positions": 0, "positive_rate": 0.0}
            continue
        scores = np.concatenate(collected[head])
        labels = np.concatenate(labels_collected[head])
        out[head] = {
            "auc": round(roc_auc(labels, scores), 5),
            "positions": int(labels.size),
            "positive_rate": round(float(labels.mean()), 5),
        }
    return out


def evaluate_discrimination(
    ranker: HSTURanker,
    split: SequenceSplit,
    vocab_size: int,
    num_negatives: int = 99,
    max_users: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Can the ranker put the true next item above random negatives?

    This is the diagnostic that separates two failures a re-ranking number
    cannot tell apart: a ranker with no signal, and a ranker with signal whose
    score blend is broken. It bypasses the blend by reporting each head
    separately, and it compares against an explicit random baseline rather
    than a vibe.

    Worth running before trusting any re-ranking result. Finding this out by
    writing it ad hoc once is how it ended up in the module.
    """
    rng = np.random.default_rng(seed)
    per_signal: dict[str, list[int]] = {name: [] for name in (*HEADS, "blended")}
    user_ids: list[str] = []

    for index in range(len(split)):
        if len(user_ids) >= max_users:
            break
        history = split.inputs[index]
        valid = history[(history > 0) & (history < vocab_size)]
        if valid.size < 3:
            continue
        prefix, positive = valid[:-1], int(valid[-1])

        negatives = rng.integers(1, vocab_size, size=num_negatives)
        negatives = negatives[negatives != positive]
        candidates = np.concatenate([[positive], negatives])

        scores = ranker.score_candidates(
            prefix,
            split.actions[index][: prefix.size],
            split.timestamps[index][: prefix.size],
            split.returned[index][: prefix.size] if index < len(split.returned) else None,
            candidates,
        )
        for name, values in scores.as_dict().items():
            # Rank of the positive, which sits at index 0.
            rank = int((values > values[0]).sum()) + 1
            per_signal[name].append(rank)
        user_ids.append(str(split.user_ids[index]) if index < len(split.user_ids) else str(index))

    if not user_ids:
        return {"users": 0}

    pool = num_negatives + 1
    out: dict[str, Any] = {
        "users": len(user_ids),
        "pool_size": pool,
        "random_hit_at_1": round(1.0 / pool, 5),
        "random_mrr": round(float(np.mean([1 / r for r in range(1, pool + 1)])), 5),
        "signals": {},
        # Kept so two rankers can be compared with the paired test rather
        # than by reading two MRR point estimates side by side. The
        # difference between two models on one user is far smaller than the
        # difference between users, so pairing is what makes a modest
        # improvement detectable at all — the same reason `evaluation.stats`
        # exists.
        "user_ids": user_ids,
        "reciprocal_ranks": {},
    }
    for name, ranks in per_signal.items():
        if not ranks:
            continue
        array = np.asarray(ranks, dtype=np.float64)
        reciprocal = 1.0 / array
        out["signals"][name] = {
            "hit_at_1": round(float((array == 1).mean()), 5),
            "hit_at_10": round(float((array <= 10).mean()), 5),
            "mrr": round(float(reciprocal.mean()), 5),
            "median_rank": float(np.median(array)),
        }
        out["reciprocal_ranks"][name] = reciprocal.tolist()
    return out


def retrieval_discrimination(
    retrieval_model: Any,
    split: SequenceSplit,
    vocab_size: int,
    num_negatives: int = 99,
    max_users: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """The same diagnostic, run on the retrieval model.

    This is the comparison that decides whether a ranker is worth wiring in.
    A second stage that orders candidates *worse* than the first stage is a
    regression with extra latency, however good its AUC looks.
    """
    rng = np.random.default_rng(seed)
    ranks: list[int] = []
    user_ids: list[str] = []

    for index in range(len(split)):
        if len(ranks) >= max_users:
            break
        history = split.inputs[index]
        valid = history[(history > 0) & (history < vocab_size)]
        if valid.size < 3:
            continue
        prefix, positive = valid[:-1], int(valid[-1])

        negatives = rng.integers(1, vocab_size, size=num_negatives)
        negatives = negatives[negatives != positive]
        candidates = np.concatenate([[positive], negatives])

        from retailgr.io.loaders import HistoryBatch

        batch = HistoryBatch(
            tokens=[prefix],
            actions=[split.actions[index][: prefix.size]],
            timestamps=[split.timestamps[index][: prefix.size]],
        )
        scores = retrieval_model.score(batch)[0][candidates]
        ranks.append(int((scores > scores[0]).sum()) + 1)
        user_ids.append(str(split.user_ids[index]) if index < len(split.user_ids) else str(index))

    if not ranks:
        return {"users": 0}
    array = np.asarray(ranks, dtype=np.float64)
    reciprocal = 1.0 / array
    return {
        "users": len(ranks),
        "pool_size": num_negatives + 1,
        "hit_at_1": round(float((array == 1).mean()), 5),
        "hit_at_10": round(float((array <= 10).mean()), 5),
        "mrr": round(float(reciprocal.mean()), 5),
        "median_rank": float(np.median(array)),
        # Same users, same order, same seed as `evaluate_discrimination`, so
        # a ranker can be tested against retrieval per user rather than mean
        # against mean.
        "user_ids": user_ids,
        "reciprocal_ranks": reciprocal.tolist(),
    }


# -- calibration --------------------------------------------------------------


def collect_slate_probabilities(
    ranker: HSTURanker,
    retrieval_model: Any,
    split: SequenceSplit,
    vocab_size: int,
    candidate_k: int = 300,
    max_users: int = 400,
) -> dict[str, dict[str, np.ndarray]]:
    """Head probabilities and outcomes on slates shaped the way serving's are.

    This function exists because the obvious alternative is wrong. The
    candidate-position probabilities the ranker learned during training were
    fitted against one true item and ``num_negatives`` uniform-random ones, so
    their base rate is 1/(1 + N) by construction — a property of the sampler,
    not of retail. Serving shows retrieval's top-k instead: a much harder set,
    drawn from a completely different distribution. Fitting a calibrator on
    the training mixture and applying it at serving time would correct one
    distribution's bias and apply it to another's.

    So the slate here is built the way the request path builds it: retrieval's
    top ``candidate_k`` for a real prefix, with items already in the history
    excluded exactly as ``exclude_seen`` excludes them at serving time. The
    positives are the customer's held-out future — the same target set the
    re-ranking metric uses — and they are *not* forced into the slate. If
    retrieval missed them, that user contributes only negatives, which is
    precisely the event that makes the true rate lower than a forced-positive
    setup would suggest.

    **The seen-exclusion is load-bearing and it is worth knowing why.** An
    earlier version used the customer's immediate next event as the positive,
    which is what ``evaluate_discrimination`` does. On this dataset that
    produced a base rate of exactly zero, for every head, on every slate —
    not a bug: 100% of click, cart and purchase events land on an item the
    customer had already interacted with, so ``exclude_seen`` removes every
    positive before the ranker sees it. Targets over the whole held-out
    window are the honest fix, and the fact that the immediate-next event is
    unmeasurable under the serving policy is itself reported.

    Returns ``{head: {"labels": ..., "probabilities": ...}}``. The ``return``
    head has no slate labels at all — a return outcome exists only for a
    matured purchase, which the target arrays do not carry — so it is left
    empty here and falls back to history positions.
    """
    from retailgr.evaluation.metrics import rank_from_scores
    from retailgr.io.loaders import HistoryBatch

    collected: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"labels": [], "probabilities": []} for name in HEADS
    }
    users = 0
    unreachable_positives = 0
    reachable_positives = 0

    for index in range(len(split)):
        if users >= max_users:
            break
        history = split.inputs[index]
        valid_mask = (history > 0) & (history < vocab_size)
        valid = history[valid_mask]
        if valid.size < 2 or index >= len(split.targets):
            continue
        targets = split.targets[index]
        if targets.size == 0:
            continue
        target_actions = (
            split.target_actions[index]
            if index < len(split.target_actions)
            and split.target_actions[index].size == targets.size
            else np.zeros(targets.size, dtype=np.int64)
        )

        actions = split.actions[index]
        row_actions = (
            actions[valid_mask]
            if actions.size == history.size
            else np.zeros(valid.size, dtype=np.int64)
        )
        timestamps = split.timestamps[index]
        row_times = (
            timestamps[valid_mask]
            if timestamps.size == history.size
            else np.arange(valid.size, dtype=np.int64)
        )
        returned_row = (
            split.returned[index][valid_mask]
            if index < len(split.returned) and split.returned[index].size == history.size
            else None
        )

        batch = HistoryBatch(tokens=[valid], actions=[row_actions], timestamps=[row_times])
        scores = retrieval_model.score(batch)[0]
        candidates = rank_from_scores(scores, candidate_k, valid)
        if candidates.size == 0:
            continue

        # How much of the future the serving policy has already ruled out.
        seen = set(valid.tolist())
        unreachable_positives += sum(1 for token in targets.tolist() if token in seen)
        reachable_positives += sum(1 for token in targets.tolist() if token not in seen)

        ranked = ranker.score_candidates(
            valid, row_actions, row_times, returned_row, candidates
        )
        # Explicitly the model's own output, not whatever calibrator may
        # already be attached: measuring "before" through an existing
        # correction would report the correction's own fit back to itself.
        probabilities = ranked.raw if ranked.raw is not None else ranked.as_dict()

        head_labels = slate_head_labels(candidates, targets, target_actions)
        for head in HEADS:
            if head == "return":
                continue
            collected[head]["labels"].append(head_labels[head])
            collected[head]["probabilities"].append(probabilities[head].astype(np.float64))
        users += 1

    out: dict[str, dict[str, np.ndarray]] = {}
    for head, arrays in collected.items():
        if not arrays["labels"]:
            out[head] = {"labels": np.zeros(0), "probabilities": np.zeros(0), "users": 0}
            continue
        out[head] = {
            "labels": np.concatenate(arrays["labels"]),
            "probabilities": np.concatenate(arrays["probabilities"]),
            "users": users,
        }
    total_positives = reachable_positives + unreachable_positives
    out["_coverage"] = {  # type: ignore[assignment]
        "users": users,
        "target_items": total_positives,
        "reachable": reachable_positives,
        "excluded_as_seen": unreachable_positives,
        "excluded_share": (
            round(unreachable_positives / total_positives, 4) if total_positives else None
        ),
    }
    return out


def slate_head_labels(
    candidates: np.ndarray, targets: np.ndarray, target_actions: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-head 0/1 labels for one slate.

    Shared by the ranker's calibration and retrieval's control measurement so
    the two resolutions are comparable. Resolution is a property of a score
    *and its label*; measuring retrieval against "any target" and the ranker
    against "a target the customer purchased" would compare two different
    questions and invite the wrong conclusion from the difference.
    """
    positive_actions = {
        head: {ACTION_TO_ID[name] for name in names}
        for head, names in HEAD_POSITIVE_ACTIONS.items()
    }
    pairs = list(zip(targets.tolist(), target_actions.tolist(), strict=True))
    out: dict[str, np.ndarray] = {}
    for head, wanted in positive_actions.items():
        qualifying = {int(token) for token, action in pairs if action in wanted}
        out[head] = np.fromiter(
            (1.0 if int(token) in qualifying else 0.0 for token in candidates),
            dtype=np.float64,
            count=candidates.size,
        )
    everything = {int(token) for token, _ in pairs}
    out["any"] = np.fromiter(
        (1.0 if int(token) in everything else 0.0 for token in candidates),
        dtype=np.float64,
        count=candidates.size,
    )
    return out


def retrieval_slate_resolution(
    retrieval_model: Any,
    split: SequenceSplit,
    vocab_size: int,
    candidate_k: int = 300,
    max_users: int = 400,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, Any]:
    """How much retrieval's *own* score explains on its *own* slate.

    The control the ranker experiment was missing. "The ranker explains 0.02%
    of the outcome variance on a slate" is only damning if something else can
    do better on the same slate, and retrieval is the obvious something else:
    it produced the slate, so if even its own ordering within that slate
    carries no information, the task is hard rather than the ranker bad.

    Retrieval emits scores, not probabilities, so they are mapped to their
    pooled rank percentile first. That is a monotone transform, and quantile
    bins are invariant to monotone transforms, so the resolution reported here
    is exactly the resolution of the raw scores — the mapping only makes them
    land in [0, 1] where the shared machinery can read them.
    """
    from retailgr.evaluation.metrics import rank_from_scores
    from retailgr.io.loaders import HistoryBatch

    labels: dict[str, list[np.ndarray]] = {}
    scores: list[np.ndarray] = []
    users = 0

    for index in range(len(split)):
        if users >= max_users:
            break
        history = split.inputs[index]
        valid_mask = (history > 0) & (history < vocab_size)
        valid = history[valid_mask]
        if valid.size < 2 or index >= len(split.targets):
            continue
        targets = split.targets[index]
        if targets.size == 0:
            continue
        target_actions = (
            split.target_actions[index]
            if index < len(split.target_actions)
            and split.target_actions[index].size == targets.size
            else np.zeros(targets.size, dtype=np.int64)
        )

        actions = split.actions[index]
        row_actions = (
            actions[valid_mask]
            if actions.size == history.size
            else np.zeros(valid.size, dtype=np.int64)
        )
        times = split.timestamps[index]
        row_times = (
            times[valid_mask]
            if times.size == history.size
            else np.arange(valid.size, dtype=np.int64)
        )

        batch = HistoryBatch(tokens=[valid], actions=[row_actions], timestamps=[row_times])
        all_scores = retrieval_model.score(batch)[0]
        candidates = rank_from_scores(all_scores, candidate_k, valid)
        if candidates.size == 0:
            continue

        for head, values in slate_head_labels(candidates, targets, target_actions).items():
            labels.setdefault(head, []).append(values)
        scores.append(np.asarray(all_scores, dtype=np.float64)[candidates])
        users += 1

    if not scores:
        return {"users": 0}

    pooled_scores = np.concatenate(scores)
    # Rank percentile: monotone, so the quantile bins are unchanged.
    order = np.argsort(pooled_scores, kind="mergesort")
    percentile = np.empty(pooled_scores.size, dtype=np.float64)
    percentile[order] = np.linspace(0.0, 1.0, pooled_scores.size)

    out: dict[str, Any] = {"users": users, "candidate_k": candidate_k, "heads": {}}
    for head, arrays in labels.items():
        report = calibration_report(np.concatenate(arrays), percentile, bins=bins)
        if report is None:
            continue
        values = report.as_dict()
        # Retrieval's scores are not probabilities, so anything that compares
        # a predicted level against an observed rate is meaningless here, and
        # dropping it is cheaper than letting someone read it.
        for key in ("mean_predicted", "bias_ratio", "ece", "ece_relative", "mce", "verdict"):
            values.pop(key, None)
        out["heads"][head] = values
    return out


def collect_teacher_forced_probabilities(
    ranker: HSTURanker, split: SequenceSplit, batch_size: int = 64, max_users: int = 1000
) -> dict[str, dict[str, np.ndarray]]:
    """The other distribution: "this item is in the history, what happened?"

    Kept alongside the slate set for one reason — the ``return`` head. A
    return outcome exists only for a matured purchase, and a slate yields at
    most one of those per user, so the serving-shaped set is usually far too
    small to fit anything on. History positions have thousands. They are a
    different distribution and the report says so rather than quietly
    substituting one for the other.
    """
    import torch

    collected: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"labels": [], "probabilities": []} for name in HEADS
    }
    usable = [
        index
        for index, history in enumerate(split.inputs)
        if ((history > 0) & (history < ranker.vocab_size)).sum() >= 2
    ][:max_users]

    ranker.net.eval()
    with torch.no_grad():
        for start in range(0, len(usable), batch_size):
            indices = usable[start : start + batch_size]
            batch = ranker._padded_history(split, indices)
            logits, _ = ranker.net(
                batch["tokens"].to(ranker.device),
                batch["actions"].to(ranker.device),
                batch["timestamps"].to(ranker.device),
            )
            label_map = ranker._labels_for(
                batch["actions"].to(ranker.device), batch["returned"].to(ranker.device)
            )
            for head in HEADS:
                label, mask = label_map[head]
                keep = mask > 0
                if not bool(keep.any()):
                    continue
                probabilities = torch.sigmoid(logits[head][:, 0::2])
                collected[head]["labels"].append(label[keep].float().cpu().numpy())
                collected[head]["probabilities"].append(
                    probabilities[keep].float().cpu().numpy().astype(np.float64)
                )

    return {
        head: (
            {
                "labels": np.concatenate(arrays["labels"]),
                "probabilities": np.concatenate(arrays["probabilities"]),
                "users": len(usable),
            }
            if arrays["labels"]
            else {"labels": np.zeros(0), "probabilities": np.zeros(0), "users": 0}
        )
        for head, arrays in collected.items()
    }


def calibrate_heads(
    ranker: HSTURanker,
    retrieval_model: Any,
    fit_split: SequenceSplit,
    test_split: SequenceSplit,
    vocab_size: int,
    candidate_k: int = 300,
    max_users: int = 400,
    kind: str = "platt",
    min_positives: int = 10,
) -> dict[str, Any]:
    """Measure calibration, fit a correction, measure again.

    The order is the point. Fitting first and reporting the result would make
    calibration look like a free improvement; measuring first states how far
    off the probabilities were, which is the number that says whether the
    expected-value blend meant anything before today.

    The calibrator is fitted on ``fit_split`` (validation) and every number in
    the report is measured on ``test_split``, so "after" is not the fit
    reproducing its own training data.
    """
    fit_slates = collect_slate_probabilities(
        ranker, retrieval_model, fit_split, vocab_size, candidate_k, max_users
    )
    test_slates = collect_slate_probabilities(
        ranker, retrieval_model, test_split, vocab_size, candidate_k, max_users
    )
    fit_history = collect_teacher_forced_probabilities(ranker, fit_split)
    test_history = collect_teacher_forced_probabilities(ranker, test_split)

    calibrators: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    heads_report: dict[str, Any] = {}

    for head in HEADS:
        slate = test_slates.get(head, {})
        history = test_history.get(head, {})
        before = calibration_report(
            slate.get("labels", []), slate.get("probabilities", [])
        )
        history_report = calibration_report(
            history.get("labels", []), history.get("probabilities", [])
        )

        # Prefer the serving-shaped set. Fall back to history positions only
        # when the slate cannot support a fit — which for `return` is the
        # normal case, not an edge case.
        source = "slate"
        fit_labels = fit_slates.get(head, {}).get("labels", np.zeros(0))
        fit_probabilities = fit_slates.get(head, {}).get("probabilities", np.zeros(0))
        if float(np.sum(fit_labels)) < min_positives:
            source = "teacher_forced"
            fit_labels = fit_history.get(head, {}).get("labels", np.zeros(0))
            fit_probabilities = fit_history.get(head, {}).get("probabilities", np.zeros(0))

        calibrator = fit_calibrator(
            fit_labels, fit_probabilities, kind=kind, min_positives=min_positives
        )
        calibrators[head] = calibrator
        provenance[head] = {
            "source": source if calibrator.kind != "identity" else "none",
            "kind": calibrator.kind,
            "fit_n": int(np.asarray(fit_labels).size),
            "fit_positives": int(float(np.sum(fit_labels))),
        }

        after = None
        if calibrator.kind != "identity" and slate.get("labels", np.zeros(0)).size:
            after = calibration_report(
                slate["labels"], calibrator.apply(slate["probabilities"])
            )

        heads_report[head] = {
            "slate": before.as_dict() if before else None,
            "teacher_forced": history_report.as_dict() if history_report else None,
            "calibrated": after.as_dict() if after else None,
            "fitted": provenance[head],
        }

    return {
        "candidate_k": candidate_k,
        "users": test_slates.get("click", {}).get("users", 0),
        "coverage": test_slates.get("_coverage", {}),
        "heads": heads_report,
        "calibrators": HeadCalibrators(calibrators=calibrators, fitted_on=provenance),
    }


def compare_against_retrieval(
    ranker_discrimination: dict[str, Any],
    retrieval: dict[str, Any],
    signal: str = "blended",
    confidence: float = 0.95,
    resamples: int = 2000,
) -> PairedComparison | None:
    """Paired test of one ranker signal against retrieval, per user.

    Returns ``None`` rather than guessing when the two runs did not score the
    same users in the same order — a paired test on misaligned rows is not a
    weaker result, it is a wrong one.
    """
    ranker_values = (ranker_discrimination.get("reciprocal_ranks") or {}).get(signal)
    retrieval_values = retrieval.get("reciprocal_ranks")
    if not ranker_values or not retrieval_values:
        return None
    if ranker_discrimination.get("user_ids") != retrieval.get("user_ids"):
        return None
    return paired_comparison(
        ranker_values, retrieval_values, confidence=confidence, resamples=resamples
    )


def compare_rankers(
    a: dict[str, Any],
    b: dict[str, Any],
    signal: str = "blended",
    confidence: float = 0.95,
    resamples: int = 2000,
) -> PairedComparison | None:
    """Paired test of two rankers' discrimination results against each other."""
    values_a = (a.get("reciprocal_ranks") or {}).get(signal)
    values_b = (b.get("reciprocal_ranks") or {}).get(signal)
    if not values_a or not values_b:
        return None
    if a.get("user_ids") != b.get("user_ids"):
        return None
    return paired_comparison(
        values_a, values_b, confidence=confidence, resamples=resamples
    )


def ranker_gate(
    ranker_discrimination: dict[str, Any],
    retrieval: dict[str, Any],
    min_relative_mrr: float = 1.0,
    calibration: dict[str, Any] | None = None,
    max_bias_ratio: float = 4.0,
) -> dict[str, Any]:
    """Decide whether the ranker may be wired into the request path.

    Two independent ways to fail, because a two-stage ranker has two
    independent ways to be wrong:

    1. **Ordering.** The blended score must order candidates at least as well
       as retrieval already does, on the same task and the same users.
       Anything less is a regression that also costs milliseconds.
    2. **Scale.** ``ScoreBlend`` is an expected value, so it multiplies each
       head's probability by a business figure. If a head is off by a factor
       of twenty in absolute terms, the weights an operator configured are not
       the weights the system applies — the blend silently becomes a different
       formula. A good AUC cannot rule this out, because AUC is invariant to
       exactly the transform that causes it.

    The second check is new and it is not decoration. The one production-shaped
    bug this project has actually hit was of that kind: `purchase 1.0,
    cart 0.3` collapsed into `0.3 x cart` because the two heads' probabilities
    lived three orders of magnitude apart. Nothing in the old gate could see
    it.

    Both fail closed: an unmeasurable criterion blocks.
    """
    ranker_mrr = (ranker_discrimination.get("signals", {}).get("blended") or {}).get("mrr")
    retrieval_mrr = retrieval.get("mrr")
    if ranker_mrr is None or retrieval_mrr is None:
        return {
            "passed": False,
            "reason": "the gate could not be evaluated; ranker left out of the bundle",
        }

    ratio = ranker_mrr / retrieval_mrr if retrieval_mrr else float("inf")
    ordering_passed = ratio >= min_relative_mrr
    verdict: dict[str, Any] = {
        "ranker_mrr": ranker_mrr,
        "retrieval_mrr": retrieval_mrr,
        "relative_mrr": round(ratio, 4),
        "min_relative_mrr": min_relative_mrr,
    }

    # Both diagnostics score the same users in the same order, so the ranker
    # can be tested against retrieval per user instead of mean against mean.
    # The ratio stays the configured rule — it is the interpretable one — but
    # a difference the paired test cannot distinguish from zero should never
    # pass a gate on the strength of a point estimate, and a significant
    # regression blocks even if the ratio somehow clears the threshold.
    comparison = compare_against_retrieval(ranker_discrimination, retrieval)
    if comparison is not None:
        verdict["paired"] = comparison.as_dict()
        verdict["paired_verdict"] = comparison.verdict()
        if comparison.significant and comparison.mean_difference < 0:
            ordering_passed = False
    verdict["ordering_passed"] = bool(ordering_passed)

    reasons: list[str] = []
    if not ordering_passed:
        detail = (
            f" ({comparison.format()} over {comparison.n} shared users)"
            if comparison is not None
            else ""
        )
        reasons.append(
            f"the ranker's MRR is {ratio:.2f}x retrieval's on the same task{detail}, "
            "so re-ranking would be a regression with extra latency"
        )

    calibration_passed = True
    if calibration is not None:
        worst_head, worst_ratio = None, 1.0
        unmeasured: list[str] = []
        # Only the heads the blend multiplies by money are checked. A head
        # that carries no business weight can be as miscalibrated as it likes.
        elsewhere: list[str] = []
        for head in ("purchase", "cart", "click", "return"):
            report = (calibration.get("heads") or {}).get(head) or {}
            # Only a measurement on the serving distribution counts. A head
            # calibrated on history positions may well be correct there; that
            # is not evidence about the slate it is applied to, and treating
            # it as evidence is the substitution this whole module exists to
            # refuse.
            measured = report.get("calibrated") or report.get("slate")
            bias = (measured or {}).get("bias_ratio")
            if bias is None or bias <= 0:
                unmeasured.append(head)
                if (report.get("fitted") or {}).get("source") == "teacher_forced":
                    elsewhere.append(head)
                continue
            distance = max(bias, 1.0 / bias)
            if distance > worst_ratio:
                worst_head, worst_ratio = head, distance
        verdict["worst_calibrated_head"] = worst_head
        verdict["worst_bias_ratio"] = round(worst_ratio, 3)
        verdict["max_bias_ratio"] = max_bias_ratio
        verdict["unmeasured_heads"] = unmeasured

        if unmeasured:
            calibration_passed = False
            detail = (
                " ("
                + ", ".join(f"`{h}`" for h in elsewhere)
                + " is calibrated on history positions, which is a different"
                " distribution from the slate it is applied to)"
                if elsewhere
                else ""
            )
            reasons.append(
                "calibration could not be measured on the serving distribution for "
                + ", ".join(f"`{h}`" for h in unmeasured)
                + detail
                + ", so the expected-value blend would be multiplying money by "
                "numbers of unknown scale"
            )
        elif worst_ratio > max_bias_ratio:
            calibration_passed = False
            reasons.append(
                f"the `{worst_head}` head is off by {worst_ratio:.1f}x in absolute "
                f"terms (limit {max_bias_ratio:.1f}x), so the configured business "
                "weights are not the weights the blend applies"
            )
    verdict["calibration_passed"] = bool(calibration_passed)

    passed = ordering_passed and calibration_passed
    verdict["passed"] = bool(passed)
    verdict["reason"] = (
        "the ranker orders candidates at least as well as retrieval"
        + (" and its probabilities are on the right scale" if calibration is not None else "")
        if passed
        else "; ".join(reasons)
    )
    return verdict


def evaluate_reranking(
    ranker: HSTURanker,
    retrieval_model: Any,
    split: SequenceSplit,
    vocab_size: int,
    candidate_k: int = 100,
    metric_k: int = 10,
    max_users: int | None = 300,
    exclude_seen: bool = True,
) -> dict[str, Any]:
    """Does reordering retrieval's candidates improve the metric?

    For each user: take retrieval's top ``candidate_k``, then score the same
    set with the ranker and reorder by the blended score. Both orderings are
    measured against the same targets at the same ``metric_k``, so the
    difference is the ranker's contribution and nothing else.
    """
    from retailgr.evaluation.metrics import rank_from_scores

    eligible = [
        index
        for index in range(len(split))
        if split.targets[index].size > 0
        and ((split.inputs[index] > 0) & (split.inputs[index] < vocab_size)).sum() >= 2
    ]
    if max_users is not None:
        eligible = eligible[:max_users]
    if not eligible:
        return {"users": 0}

    retrieval_ndcg: list[float] = []
    reranked_ndcg: list[float] = []
    retrieval_recall: list[float] = []
    reranked_recall: list[float] = []
    changed_top1 = 0

    for index in eligible:
        history = split.inputs[index]
        targets = set(split.targets[index].tolist())

        batch = split.batch(index, index + 1)
        scores = retrieval_model.score(batch)[0]
        exclude = history if exclude_seen else ()
        candidates = rank_from_scores(scores, candidate_k, exclude)
        if candidates.size == 0:
            continue

        ranked = ranker.score_candidates(
            history,
            split.actions[index],
            split.timestamps[index],
            split.returned[index] if index < len(split.returned) else None,
            candidates,
        )
        order = np.argsort(-ranked.blended)
        reordered = candidates[order]

        retrieval_ndcg.append(ndcg_at_k(candidates.tolist(), targets, metric_k))
        reranked_ndcg.append(ndcg_at_k(reordered.tolist(), targets, metric_k))
        retrieval_recall.append(recall_at_k(candidates.tolist(), targets, metric_k))
        reranked_recall.append(recall_at_k(reordered.tolist(), targets, metric_k))
        if int(candidates[0]) != int(reordered[0]):
            changed_top1 += 1

    if not retrieval_ndcg:
        return {"users": 0}

    def mean(values: list[float]) -> float:
        clean = [value for value in values if value == value]
        return round(float(np.mean(clean)), 6) if clean else float("nan")

    before_ndcg, after_ndcg = mean(retrieval_ndcg), mean(reranked_ndcg)
    return {
        "users": len(retrieval_ndcg),
        "candidate_k": candidate_k,
        "metric_k": metric_k,
        f"retrieval_ndcg@{metric_k}": before_ndcg,
        f"reranked_ndcg@{metric_k}": after_ndcg,
        "ndcg_delta_pct": (
            round((after_ndcg - before_ndcg) / before_ndcg * 100, 2) if before_ndcg else None
        ),
        f"retrieval_recall@{metric_k}": mean(retrieval_recall),
        f"reranked_recall@{metric_k}": mean(reranked_recall),
        "top1_changed_share": round(changed_top1 / len(retrieval_ndcg), 4),
    }


def render_calibration_section(calibration: dict[str, Any]) -> str:
    """The reliability of each head, in the terms the blend cares about."""
    if not calibration or not calibration.get("heads"):
        return (
            "## Calibration\n\nNot measured. Until it is, the expected-value "
            "blend is multiplying business weights by numbers of unknown "
            "scale, and the weights an operator configures are not the "
            "weights the system applies."
        )

    lines = [
        "## Calibration: are these probabilities, or just scores?",
        "",
        f"Measured on slates the shape serving builds them — retrieval's top "
        f"{calibration.get('candidate_k')} for {calibration.get('users')} held-out "
        "users, with items already in the history excluded exactly as the "
        "request path excludes them, and the held-out future as the outcome.",
        "",
    ]
    coverage = calibration.get("coverage") or {}
    if coverage.get("excluded_share") is not None:
        lines.append(
            f"**{coverage['excluded_share']:.0%} of these customers' future "
            f"interactions ({coverage['excluded_as_seen']:,} of "
            f"{coverage['target_items']:,}) are with items already in their "
            "history, so `exclude_seen` removes them from the slate before the "
            "ranker ever scores them.** That is a property of the serving "
            "policy, not of the model, and it caps what any ranker here can be "
            "measured against. It is worth knowing separately that on this "
            "dataset *every* click, cart and purchase lands on an "
            "already-seen item — so the immediate next engagement is, under "
            "this policy, not a thing the system can be scored on at all."
        )
        lines.append("")
    header = [
        "Head",
        "Base rate",
        "Mean predicted",
        "Bias",
        "ECE / base",
        "Brier skill",
        "Resolution",
        "Verdict",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    def row(label: str, values: dict[str, Any] | None) -> str:
        if not values:
            return f"| {label} | – | – | – | – | – | – | not measured |"
        bias = values.get("bias_ratio")
        ece_relative = values.get("ece_relative")
        skill = values.get("brier_skill")
        resolution = values.get("resolution_share")
        return "| " + " | ".join(
            [
                label,
                f"{values['base_rate']:.5f}",
                f"{values['mean_predicted']:.5f}",
                f"{bias:.2f}x" if bias is not None else "–",
                f"{ece_relative:.2f}" if ece_relative is not None else "–",
                f"{skill:+.3f}" if skill is not None else "–",
                f"{resolution:.2%}" if resolution is not None else "–",
                values.get("verdict", "–"),
            ]
        ) + " |"

    for head, report in calibration["heads"].items():
        lines.append(row(f"`{head}`", report.get("slate")))
        if report.get("calibrated"):
            lines.append(row(f"`{head}` **calibrated**", report["calibrated"]))
    lines.append("")
    lines.append(
        "*Bias* is mean predicted over the base rate: 1.00x is right on "
        "average, 0.05x means the head believes the event is twenty times "
        "rarer than it is — and in an expected-value blend that is the same "
        "as dividing its business weight by twenty. *ECE / base* is the "
        "calibration error as a share of the base rate, because an absolute "
        "ECE of 0.003 is negligible for a coin flip and total for an event "
        "that happens three times in a thousand. *Brier skill* is against "
        "always predicting the base rate: **negative means the head is worse "
        "than that constant**, however good its AUC."
    )
    lines.append("")
    lines.append(
        "*Resolution* is the share of the outcome's variance the head's "
        "predictions actually explain, and it is the column to read first. "
        "Calibration cannot change it: a monotone rescaling moves where the "
        "probabilities sit, never how well they separate outcomes. So a head "
        "with resolution near zero can be made perfectly calibrated and will "
        "still be a constant wearing a probability's clothes — which is why "
        "the verdict says *calibrated but uninformative* rather than "
        "*calibrated* when that happens."
    )
    lines.append("")

    # The comparison that explains everything else on this page.
    contrast = [
        (head, report.get("teacher_forced"), report.get("slate"))
        for head, report in calibration["heads"].items()
    ]
    if any(t and s for _, t, s in contrast):
        lines.append("### The same heads, on the two distributions")
        lines.append("")
        header = [
            "Head",
            "Resolution (history)",
            "Resolution (slate)",
            "Brier skill (history)",
            "Brier skill (slate)",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        def cell(values: dict[str, Any] | None, key: str, fmt: str) -> str:
            if not values or values.get(key) is None:
                return "–"
            return format(values[key], fmt)

        for head, history, slate in contrast:
            if not (history or slate):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{head}`",
                        cell(history, "resolution_share", ".2%"),
                        cell(slate, "resolution_share", ".2%"),
                        cell(history, "brier_skill", "+.3f"),
                        cell(slate, "brier_skill", "+.3f"),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append(
            "Left columns: *given this item is in the customer's history, "
            "which action did they take on it?* Right columns: *given this "
            "item is in a slate we are about to show, will they act on it?* "
            "The heads are trained on the first and deployed on the second. "
            "Every AUC in this report, and every AUC in the literature this "
            "design is drawn from, is measured on the left."
        )
        lines.append("")

    fitted_rows = [
        (head, report.get("fitted") or {})
        for head, report in calibration["heads"].items()
    ]
    if any(values for _, values in fitted_rows):
        lines.append("### What was fitted")
        lines.append("")
        header = ["Head", "Calibrator", "Fitted on", "Examples", "Positives"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for head, values in fitted_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{head}`",
                        f"`{values.get('kind', 'identity')}`",
                        f"`{values.get('source', 'none')}`",
                        f"{values.get('fit_n', 0):,}",
                        f"{values.get('fit_positives', 0):,}",
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append(
            "`identity` means no correction was installed — either too few "
            "outcomes to fit on, or the fit failed its own checks (it "
            "reversed the head's ordering, saturated, or did not improve "
            "calibration on the data it was fitted to). Declining is a "
            "result the report can print; a silently broken correction "
            "applied to every request is not."
        )
        lines.append("")
        lines.append(
            "`teacher_forced` as a source is a distribution mismatch stated "
            "rather than hidden: a return outcome exists only for a matured "
            "purchase, and a slate yields at most one of those per user, so "
            "the `return` head is normally fitted on history positions and "
            "applied to candidates. That is an extrapolation, and it is the "
            "reason the return term deserves the least trust in the blend."
        )
        lines.append("")

    lines.append(
        "Calibration is monotone per head, so it cannot change any single "
        "head's AUC. It changes the **blend**, because the blend sums heads "
        "that previously lived on incomparable scales — which is the entire "
        "point and also the only thing worth measuring afterwards."
    )
    lines.append("")
    lines.append(
        "One limitation these numbers cannot escape: the positive is the item "
        "the customer actually interacted with next, and every other candidate "
        "is labelled 0 including the ones they would have liked and never saw. "
        "So a calibrated P(purchase) of 0.004 means *4 in 1000 slates like "
        "this one had this item as the next purchase*, not *4 in 1000 "
        "customers would buy it*. Closing that gap needs exposure logs."
    )
    return "\n".join(lines)


def render_ranker_report(result: dict[str, Any]) -> str:
    """Markdown report for the ranker."""
    lines = [
        "# Ranker",
        "",
        f"- Dataset: `{result.get('dataset')}`, variant `{result.get('variant')}`",
        f"- Token vocabulary: {result.get('vocab_size')}",
        f"- Fit: {result.get('fit_seconds')}s, "
        f"{int(result.get('fit', {}).get('parameters', 0)):,} parameters",
        "",
        "## Head calibration (test split)",
        "",
    ]
    header = ["Head", "AUC", "Positions", "Positive rate"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for head, values in (result.get("heads") or {}).items():
        auc = values.get("auc")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{head}`",
                    "n/a" if auc != auc else f"{auc:.4f}",
                    f"{values.get('positions', 0):,}",
                    f"{values.get('positive_rate', 0):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "AUC 0.5 is coin-flipping. A head with `n/a` saw only one class, which "
        "for `return` usually means the return window swallowed every matured "
        "purchase — check `purchases_matured` in the sequence stats."
    )
    lines.append("")
    lines.append(
        "**AUC is not calibration.** It is invariant to any monotone transform "
        "of the scores, so these numbers say the heads *order* well and say "
        "nothing about whether their probabilities are on the probability "
        "scale. The blend multiplies them by money, so the next section is the "
        "one that decides whether that multiplication means anything."
    )
    lines.append("")
    lines.append(render_calibration_section(result.get("calibration") or {}))

    gate = result.get("gate") or {}
    if gate:
        verdict = "PASSED" if gate.get("passed") else "BLOCKED"
        lines.append(f"## Gate: {verdict}")
        lines.append("")
        lines.append(gate.get("reason", ""))
        lines.append("")
        criteria = ["| Criterion | Measured | Threshold | Verdict |", "| --- | --- | --- | --- |"]
        if gate.get("relative_mrr") is not None:
            criteria.append(
                f"| Orders at least as well as retrieval "
                f"| {gate['relative_mrr']:.2f}x "
                f"| >= {gate['min_relative_mrr']:.2f}x "
                f"| {'pass' if gate.get('ordering_passed') else '**fail**'} |"
            )
        if "calibration_passed" in gate:
            worst = gate.get("worst_bias_ratio")
            head = gate.get("worst_calibrated_head")
            measured = (
                f"{worst:.1f}x off (`{head}`)"
                if worst is not None and head
                else ("not measured" if gate.get("unmeasured_heads") else "1.0x off")
            )
            criteria.append(
                f"| Probabilities on the right scale | {measured} "
                f"| <= {gate.get('max_bias_ratio', 0):.1f}x "
                f"| {'pass' if gate.get('calibration_passed') else '**fail**'} |"
            )
        if len(criteria) > 2:
            lines.extend(criteria)
            lines.append("")
            lines.append(
                f"Ranker MRR {gate['ranker_mrr']:.4f} against retrieval's "
                f"{gate['retrieval_mrr']:.4f} on the same task and the same users."
            )
        if not gate.get("passed"):
            lines.append("")
            lines.append(
                "The ranker is still written to the bundle directory so the next "
                "run has something to compare against, but it is **not** wired "
                "into the request path: the API serves retrieval order. "
                "`--force-ranker` overrides this."
            )
        lines.append("")

    discrimination = result.get("discrimination") or {}
    if discrimination.get("users"):
        pool = discrimination["pool_size"]
        lines.append(f"## Can it pick the true next item out of {pool}?")
        lines.append("")
        header = ["Signal", "hit@1", "hit@10", "MRR", "Median rank"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        lines.append(
            f"| _random_ | {discrimination['random_hit_at_1']:.4f} "
            f"| {10 / pool:.4f} | {discrimination['random_mrr']:.4f} | {pool / 2:.0f} |"
        )
        for name, values in discrimination["signals"].items():
            emphasis = "**" if name == "blended" else ""
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{emphasis}`{name}`{emphasis}",
                        f"{emphasis}{values['hit_at_1']:.4f}{emphasis}",
                        f"{emphasis}{values['hit_at_10']:.4f}{emphasis}",
                        f"{emphasis}{values['mrr']:.4f}{emphasis}",
                        f"{emphasis}{values['median_rank']:.0f}{emphasis}",
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append(
            "Each head is reported separately because the blend can destroy a "
            "signal the heads have: if a head beats random here and `blended` "
            "does not, the weights are wrong, not the model."
        )
        lines.append("")

    rerank = result.get("reranking") or {}
    if rerank.get("users"):
        metric_k = rerank["metric_k"]
        lines.append(f"## Does re-ranking beat retrieval alone? (@{metric_k})")
        lines.append("")
        header = ["Ordering", f"NDCG@{metric_k}", f"Recall@{metric_k}"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        lines.append(
            f"| retrieval only | {rerank[f'retrieval_ndcg@{metric_k}']:.4f} "
            f"| {rerank[f'retrieval_recall@{metric_k}']:.4f} |"
        )
        delta = rerank.get("ndcg_delta_pct")
        lines.append(
            f"| **+ ranker** | **{rerank[f'reranked_ndcg@{metric_k}']:.4f}"
            + (f" ({delta:+.1f}%)**" if delta is not None else "**")
            + f" | {rerank[f'reranked_recall@{metric_k}']:.4f} |"
        )
        lines.append("")
        lines.append(
            f"Same {rerank['candidate_k']} candidates, same targets, "
            f"{rerank['users']} users — only the ordering differs. The ranker "
            f"changed the top result for {rerank['top1_changed_share']:.0%} of them."
        )
        lines.append("")
        lines.append(
            "Recall@k is unchanged by construction when `candidate_k` is the "
            "size of the pool being reordered and k equals it; a difference "
            "here means the ranker moved relevant items into the top k."
        )
    return "\n".join(lines)
