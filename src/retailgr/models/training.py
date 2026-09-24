"""Early stopping on the validation split, which every model accepted and ignored.

Every `fit` in this repository has the signature `fit(train, val)`, and every
one of the nine call sites passes `data.val`. **None of the four
implementations read it.** HSTU and SASRec trained a fixed eight epochs and
returned; the ranker the same; popularity has no epochs to stop. So every
comparison this project reported — HSTU tied with SASRec at p=0.225, each
ablation finding — was measured at an arbitrary stopping point, and the
interface said otherwise to anyone reading a call site.

What this adds is ordinary and the details are where it can go wrong:

**Selection is on validation, never on test.** The metric is computed on
`val` after every epoch, the best state is kept, and the model is restored to
it. Test is only ever touched by the harness afterwards. Early-stopping on
test would report the best of N looks at the held-out set as if it were one,
and `tests/test_early_stopping.py` asserts structurally that no harness does.

**The validation metric is computed the way the test metric is.** Same
`exclude_seen`, same `k`. A model that early-stops on recall with seen items
included and is then reported with them excluded has been selected for a
different objective from the one it is judged on.

**The network is put back in training mode after every check.** `evaluate`
calls `score`, which calls `net.eval()`. Forgetting to switch back is silent
and it is not small: dropout stays off for every remaining epoch, so the
model after the first validation pass trains as a different model from the
one before it. It is the easiest bug in this file to write and the hardest to
notice, so it has its own test.

**Restoring the best state is the point, not a detail.** With patience 5 the
loop runs five epochs past the best one before it stops; returning the final
weights would report a model that validation had already rejected.

Both HSTU and SASRec get the identical rule — same metric, same patience,
same cap — because the comparison between them is the thing being fixed.
Giving each model the epochs it needs is fairer than giving both the same
arbitrary number; giving them *different* stopping rules would be the least
fair option of the three.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["EarlyStopping", "EarlyStoppingConfig", "validation_metric"]


@dataclass
class EarlyStoppingConfig:
    """How long to train, decided by the validation split.

    Absent from a model config means *off*, which keeps the old fixed-epoch
    behaviour for every caller that has not opted in — tiny models in unit
    tests above all, which would otherwise run to `max_epochs`.
    """

    enabled: bool = False
    metric: str = "ndcg@10"
    patience: int = 5
    max_epochs: int = 200
    # An improvement smaller than this does not reset patience. Zero means
    # any strict improvement counts; the default stays there because a
    # threshold picked by taste is a second hyperparameter nobody tunes.
    min_delta: float = 0.0
    # Must match how the test split is scored. The harness injects the
    # pipeline's `evaluation.exclude_seen` so the two cannot disagree.
    exclude_seen: bool = True
    # Validation users per check. None is all of them; a cap trades a
    # noisier curve for faster epochs on a large split.
    max_users: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> EarlyStoppingConfig:
        block = dict((config or {}).get("early_stopping") or {})
        unknown = set(block) - set(cls.__dataclass_fields__)
        if unknown:
            # A typo like `patiance` must not silently fall back to the
            # default — that is a run that believes it tuned something.
            raise ValueError(f"unknown early_stopping keys: {sorted(unknown)}")
        return cls(**block)

    @property
    def k(self) -> int:
        _, _, value = self.metric.partition("@")
        if not value.isdigit():
            raise ValueError(f"early_stopping.metric must look like 'ndcg@10', got {self.metric!r}")
        return int(value)


def validation_metric(model: Any, val: Any, settings: EarlyStoppingConfig) -> float:
    """The number early stopping watches, computed exactly as test is."""
    from retailgr.evaluation.evaluate import evaluate
    from retailgr.io.loaders import SequenceSplit

    split = val
    if settings.max_users is not None and len(val) > settings.max_users:
        split = SequenceSplit(
            user_ids=val.user_ids[: settings.max_users],
            inputs=val.inputs[: settings.max_users],
            targets=val.targets[: settings.max_users],
            actions=val.actions[: settings.max_users],
            timestamps=val.timestamps[: settings.max_users],
            returned=val.returned[: settings.max_users],
            target_actions=val.target_actions[: settings.max_users]
            if val.target_actions
            else [],
        )
    result = evaluate(
        model,
        split,
        vocab_size=model.vocab_size,
        k_values=[settings.k],
        exclude_seen=settings.exclude_seen,
        with_intervals=False,
    )
    overall = result.get("overall", {})
    if settings.metric not in overall:
        raise ValueError(
            f"validation produced no {settings.metric!r}; available: {sorted(overall)}"
        )
    return float(overall[settings.metric])


@dataclass
class EarlyStopping:
    """Watches the validation metric, keeps the best weights, says when to stop."""

    settings: EarlyStoppingConfig
    best_value: float = float("-inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    curve: list[float] = field(default_factory=list)
    seconds_in_validation: float = 0.0
    stopped_because: str = "max_epochs"
    _best_state: dict[str, Any] | None = None

    def check(self, epoch: int, model: Any, net: Any, val: Any) -> bool:
        """Score this epoch, keep it if it is the best so far. True means stop."""
        started = time.time()
        value = validation_metric(model, val, self.settings)
        self.seconds_in_validation += time.time() - started
        self.curve.append(value)

        if value > self.best_value + self.settings.min_delta:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            # A deep copy, not a reference: `state_dict()` returns the live
            # tensors, and the next optimiser step would overwrite the "best"
            # weights in place.
            self._best_state = copy.deepcopy(net.state_dict())
        else:
            self.epochs_without_improvement += 1

        # `evaluate` -> `score` -> `net.eval()`. Without this line dropout is
        # off for every remaining epoch and the model silently trains as a
        # different network after its first validation pass.
        net.train()

        if self.epochs_without_improvement >= self.settings.patience:
            self.stopped_because = "patience"
            return True
        return False

    def restore(self, net: Any) -> None:
        """Put the best weights back. Returning the last ones would report a
        model that validation had already rejected `patience` epochs ago."""
        if self._best_state is not None:
            net.load_state_dict(self._best_state)
        net.eval()

    def summary(self, epochs_run: int) -> dict[str, Any]:
        return {
            "early_stopping": True,
            "val_metric": self.settings.metric,
            "best_epoch": self.best_epoch + 1,  # 1-based, as a person reads it
            "best_val": round(self.best_value, 6),
            "epochs_run": epochs_run,
            "stopped_because": self.stopped_because,
            "val_curve": [round(value, 6) for value in self.curve],
            "seconds_in_validation": round(self.seconds_in_validation, 1),
        }
