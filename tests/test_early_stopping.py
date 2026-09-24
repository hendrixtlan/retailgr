"""Early stopping on the validation split every model accepted and ignored.

`fit(train, val)` was the signature of all four models and nine call sites
passed `data.val`. None of the implementations read it, so every reported
comparison — HSTU tied with SASRec, each ablation — was taken at a fixed,
arbitrary eight epochs while the call sites suggested validation was
steering training.

These tests pin the four ways an early-stopping implementation goes quietly
wrong:

1. **Dropout left off after the first check.** `evaluate` puts the network
   in eval mode; forgetting to switch back means every later epoch trains a
   different model. Invisible in any loss curve.
2. **The final weights returned instead of the best ones.** With patience
   five, that is a model validation rejected five epochs ago.
3. **A reference kept instead of a copy.** `state_dict()` returns live
   tensors; the next optimiser step overwrites the "best" weights in place.
4. **Selection on test.** The best of N looks at the held-out set, reported
   as one.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch

from retailgr.io.loaders import SequenceSplit
from retailgr.models.training import EarlyStopping, EarlyStoppingConfig


def _split(n_users: int = 40, length: int = 12, vocab: int = 30, seed: int = 0) -> SequenceSplit:
    """Sequences with learnable structure: each user cycles through a few
    items, so a model can genuinely improve and then overfit."""
    rng = np.random.default_rng(seed)
    split = SequenceSplit()
    for user in range(n_users):
        base = rng.integers(1, vocab - 4)
        tokens = np.array([base + (i % 4) for i in range(length)], dtype=np.int64)
        split.user_ids.append(f"U{user}")
        split.inputs.append(tokens[:-2])
        split.targets.append(tokens[-2:])
        split.actions.append(np.ones(length - 2, dtype=np.int64))
        split.timestamps.append(np.arange(length - 2, dtype=np.int64))
        split.returned.append(np.full(length - 2, -1, dtype=np.int64))
        split.target_actions.append(np.ones(2, dtype=np.int64))
    return split


def _model(kind: str, **early):
    from retailgr.models.hstu import HSTUModel
    from retailgr.models.sasrec import SASRecModel

    config = {
        "hidden_dim": 16, "num_blocks": 1, "num_heads": 1, "max_len": 12,
        "batch_size": 16, "epochs": 3, "seed": 5, "dropout": 0.3,
    }
    if early:
        config["early_stopping"] = {"enabled": True, **early}
    return (HSTUModel if kind == "hstu" else SASRecModel)(30, config)


# -- configuration ------------------------------------------------------------


def test_absent_means_off_so_existing_callers_keep_fixed_epochs():
    """Unit tests build tiny models and pass a val split; on by default they
    would each run to `max_epochs`."""
    assert EarlyStoppingConfig.from_config({}).enabled is False
    assert EarlyStoppingConfig.from_config(None).enabled is False


def test_a_typo_in_the_block_is_an_error_not_a_default():
    """`patiance: 3` silently falling back to 5 is a run that believes it
    tuned something."""
    with pytest.raises(ValueError, match="unknown early_stopping keys"):
        EarlyStoppingConfig.from_config({"early_stopping": {"patiance": 3}})


def test_the_metric_must_name_a_cutoff():
    with pytest.raises(ValueError, match="ndcg@10"):
        _ = EarlyStoppingConfig(metric="ndcg").k


@pytest.mark.parametrize("kind", ["hstu", "sasrec"])
def test_without_early_stopping_the_loop_is_exactly_what_it_was(kind):
    model = _model(kind)
    stats = model.fit(_split(), _split(seed=1))
    assert stats["early_stopping"] is False
    assert stats["epochs"] == 3.0
    assert len(stats["train_loss_curve"]) == 3


# -- the loop -----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["hstu", "sasrec"])
def test_it_stops_on_patience_and_reports_the_curve(kind):
    model = _model(kind, patience=2, max_epochs=60, metric="ndcg@10")
    stats = model.fit(_split(), _split(seed=1))
    assert stats["early_stopping"] is True
    assert len(stats["val_curve"]) == stats["epochs_run"]
    assert 1 <= stats["best_epoch"] <= stats["epochs_run"]
    if stats["stopped_because"] == "patience":
        # Exactly `patience` epochs past the best, no more.
        assert stats["epochs_run"] - stats["best_epoch"] == 2


@pytest.mark.parametrize("kind", ["hstu", "sasrec"])
def test_the_returned_model_is_the_best_one_not_the_last(kind):
    """The whole point of keeping a checkpoint. Measured on the validation
    split: the restored model must score exactly the recorded best."""
    from retailgr.models.training import validation_metric

    val = _split(seed=1)
    model = _model(kind, patience=3, max_epochs=40, metric="ndcg@10")
    stats = model.fit(_split(), val)
    rescored = validation_metric(model, val, model.early_stopping)
    assert rescored == pytest.approx(stats["best_val"], abs=1e-6)
    assert rescored == pytest.approx(max(stats["val_curve"]), abs=1e-6)


def test_the_network_is_back_in_training_mode_after_every_check():
    """The bug this file most exists for.

    `evaluate` -> `score` -> `net.eval()`. Without switching back, dropout is
    off for every remaining epoch and the model trains as a different
    network after its first validation pass — nothing in the loss curve says
    so.
    """
    model = _model("hstu", patience=100, max_epochs=3)
    monitor = EarlyStopping(model.early_stopping)
    model.net.train()
    monitor.check(0, model, model.net, _split(seed=1))
    assert model.net.training, "check() left the network in eval mode"
    assert all(m.training for m in model.net.modules()), "a submodule stayed in eval mode"


def test_the_best_weights_are_a_copy_not_a_live_reference():
    """`state_dict()` returns the live tensors. Keeping a reference means the
    next optimiser step overwrites the "best" weights in place, and restore
    hands back the last epoch while claiming the best."""
    model = _model("hstu", patience=100, max_epochs=3)
    monitor = EarlyStopping(model.early_stopping)
    monitor.check(0, model, model.net, _split(seed=1))
    saved = {k: v.clone() for k, v in monitor._best_state.items()}

    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.add_(1.0)

    for key, value in monitor._best_state.items():
        assert torch.equal(value, saved[key]), f"{key} changed with the live weights"


def test_restore_leaves_the_network_ready_to_serve():
    model = _model("hstu", patience=100, max_epochs=3)
    monitor = EarlyStopping(model.early_stopping)
    monitor.check(0, model, model.net, _split(seed=1))
    monitor.restore(model.net)
    assert not model.net.training


def test_min_delta_stops_tiny_improvements_from_resetting_patience():
    settings = EarlyStoppingConfig(enabled=True, patience=2, min_delta=0.01)
    monitor = EarlyStopping(settings)

    class _Net:
        def state_dict(self):
            return {"w": torch.zeros(1)}

        def train(self):
            return None

    values = iter([0.10, 0.105, 0.109, 0.108])
    import retailgr.models.training as training

    original = training.validation_metric
    training.validation_metric = lambda *a, **k: next(values)
    try:
        assert monitor.check(0, None, _Net(), None) is False  # 0.10 is the best
        assert monitor.check(1, None, _Net(), None) is False  # +0.005 < min_delta
        assert monitor.check(2, None, _Net(), None) is True  # +0.009, still not enough
    finally:
        training.validation_metric = original
    assert monitor.best_epoch == 0


# -- selection happens on validation, never on test ---------------------------


def _fit_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    ]


def test_no_harness_hands_the_test_split_to_fit():
    """Selecting on test reports the best of N looks at the held-out set as
    if it were one. Checked on every call site, structurally, because the
    mistake is a one-word edit that every behavioural test would pass."""
    from pathlib import Path

    offenders = []
    for path in sorted(Path("src/retailgr").rglob("*.py")):
        for call in _fit_calls(path.read_text(encoding="utf-8")):
            for argument in list(call.args) + [kw.value for kw in call.keywords]:
                text = ast.unparse(argument)
                if "test" in text.split("."):
                    offenders.append(f"{path}:{call.lineno} fit({text})")
    assert not offenders, offenders


def test_the_validation_metric_uses_the_pipelines_exclude_seen():
    """A model selected with seen items included and reported with them
    excluded was tuned for a different objective from the one it is judged
    on. The harness injects the pipeline setting; this checks it lands."""
    from retailgr.experiment import build_model

    config = {"type": "hstu", "hidden_dim": 8, "max_len": 8,
              "early_stopping": {"enabled": True, "exclude_seen": True}}
    _, model = build_model(30, config, exclude_seen=False)
    assert model.early_stopping.exclude_seen is False
    # And the caller's dict is not mutated along the way.
    assert config["early_stopping"]["exclude_seen"] is True


def test_the_defaults_agree_with_the_evaluation_default():
    from pathlib import Path

    import yaml

    pipeline = yaml.safe_load(Path("configs/pipeline.yaml").read_text(encoding="utf-8"))
    assert EarlyStoppingConfig().exclude_seen == bool(pipeline["evaluation"]["exclude_seen"])


def test_every_fit_call_in_the_harness_passes_a_validation_split():
    """The call sites that build compared models must actually give the
    model something to stop on — otherwise early stopping is configured,
    silently skipped, and the comparison is back to a fixed epoch count."""
    from retailgr import experiment

    source = inspect.getsource(experiment)
    calls = _fit_calls(source)
    assert calls, "found no fit() calls in experiment.py"
    for call in calls:
        texts = [ast.unparse(a) for a in call.args] + [ast.unparse(k.value) for k in call.keywords]
        assert any("val" in text for text in texts), f"line {call.lineno}: fit({texts})"


# -- the shipped configurations -----------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["hstu_small", "sasrec_small", "hstu_base", "sasrec_base", "hstu_ml1m", "sasrec_ml1m"],
)
def test_every_shipped_retrieval_config_trains_until_validation_says_stop(name):
    """On the synthetic data this changes nothing measurable — converged
    against eight epochs is -0.0012 (p=0.401) for SASRec and -0.0001
    (p=0.845) for HSTU. It is on because the fixed count was tuned to
    nothing and happened to land on this dataset's plateau; the next dataset
    will not be this one."""
    from retailgr.config import load_model_config

    settings = EarlyStoppingConfig.from_config(load_model_config(f"{name}.yaml"))
    assert settings.enabled, f"{name} trains a fixed schedule"
    assert settings.metric == "ndcg@10"
    assert settings.max_epochs >= 50, "a cap this low is a fixed schedule in disguise"


def test_the_ranker_is_known_not_to_early_stop():
    """Stated rather than hidden. The ranker's `fit` still ignores `val`: its
    natural criterion is slate AUC, which needs retrieval's candidates for
    every validation user on every epoch. Until that is paid for, the ranker
    trains a fixed schedule — and this test fails the day that changes, so
    the README's claim gets updated with it."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/retailgr/models/ranker.py").read_text(encoding="utf-8"))
    fit = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "fit"
    )
    reads_val = any(isinstance(n, ast.Name) and n.id == "val" for n in ast.walk(fit))
    assert not reads_val, "the ranker now reads val; update the README and remove this test"


# -- the ablation report names its cutoff -------------------------------------


def _ablation_result() -> dict:
    stats = {
        "mean_a": 0.1, "mean_b": 0.1, "mean_difference": 0.001, "ci_low": -0.001,
        "ci_high": 0.003, "relative_difference": 0.01, "p_value": 0.4, "n": 900,
        "significant": False, "confidence": 0.95,
    }
    per_cutoff = {"ndcg@10": dict(stats), "ndcg@200": {**stats, "p_value": 0.001,
                  "ci_low": 0.002, "significant": True}}
    return {
        "variant": "config", "vocab_size": 945, "k_values": [10, 50, 200],
        "base_model_config": "hstu_small.yaml", "confidence": 0.95, "seeds": [13],
        "models": {"hstu_full": {"type": "hstu", "fit": {"final_epoch_loss": 3.5},
                                 "test": {"overall": {"ndcg@200": 0.18}, "intervals": {}}}},
        "comparisons": {"vs_hstu_full": {"hstu_no_rab": per_cutoff}},
    }


def test_every_verdict_table_is_titled_with_its_cutoff():
    """The defect this guards: the ablation rendered only NDCG@200, in a
    table whose column was called "Difference", and the README quoted it as
    "the temporal bias costs 3.7%". At NDCG@10 — the ten items a customer
    sees — the same comparison was +0.0003, p=0.740."""
    from retailgr.experiment import render_ablation

    text = render_ablation(_ablation_result())
    verdict_titles = [line for line in text.splitlines() if "paired on the same users" in line]
    assert verdict_titles, "no verdict tables rendered"
    for title in verdict_titles:
        assert "NDCG@" in title, f"a verdict table has no cutoff: {title!r}"


def test_the_top_ten_is_reported_alongside_the_deep_cutoff():
    """Both, always. A component that matters at @200 and not at @10 is a
    finding about the tail of a long list, and the report has to make that
    visible rather than leave it to whoever copies the table."""
    from retailgr.experiment import render_ablation

    text = render_ablation(_ablation_result())
    assert "NDCG@10, paired" in text
    assert "NDCG@200, paired" in text
    # And a significant verdict carries its cutoff in the verdict itself, so
    # quoting the cell alone still quotes the metric.
    assert "better at @200" in text


def test_the_ablation_report_says_how_many_users_it_is_over():
    """A number without its dataset is how the README quoted 1,101 test users
    for weeks after the pipeline had started producing 943."""
    from retailgr.experiment import render_ablation

    result = _ablation_result()
    result["models"]["hstu_full"]["test"]["overall"]["eval_users"] = 943.0
    assert "- Test users: 943" in render_ablation(result)


def test_restore_puts_back_the_best_checkpoint_not_the_current_weights():
    """The version of "returns the best, not the last" that can fail.

    `test_the_returned_model_is_the_best_one_not_the_last` above passed with
    `restore` disabled outright — the mutation check caught it. On a fixture
    small enough to learn completely, the validation metric saturates, the
    last epoch scores exactly what the best one did, and returning either
    looks the same. So this drives the monitor directly: a best epoch, then
    different weights and a worse score, then restore — and asserts the
    weights that come back are the best epoch's, not the ones in place.
    """
    import retailgr.models.training as training

    model = _model("hstu", patience=100, max_epochs=3)
    monitor = EarlyStopping(model.early_stopping)
    scores = iter([0.50, 0.10])
    original = training.validation_metric
    training.validation_metric = lambda *a, **k: next(scores)
    try:
        monitor.check(0, model, model.net, None)
        best = {k: v.clone() for k, v in model.net.state_dict().items()}
        with torch.no_grad():
            for parameter in model.net.parameters():
                parameter.mul_(-3.0)
        monitor.check(1, model, model.net, None)
    finally:
        training.validation_metric = original

    monitor.restore(model.net)
    for key, value in model.net.state_dict().items():
        assert torch.equal(value, best[key]), f"{key} is the last epoch's, not the best's"
