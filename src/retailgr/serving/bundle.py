"""The serving bundle: everything the API needs, exported from a training run.

A bundle is a directory, versioned and immutable:

    bundle/
      manifest.json      what this is, which data and code produced it
      model.pt           the trained encoder's weights
      item_embeddings.npy  the retrieval index's vectors
      vocab.json         token id -> token, category
      hierarchy.json     token -> the SKUs it resolves to

Exporting the item embeddings separately is not redundancy. Retrieval needs a
matrix to search, the ranker needs the encoder, and the two scale differently:
the embedding table can be sharded onto a vector database while the encoder
sits on a GPU. Keeping them as separate artefacts is what makes that possible
later without re-exporting.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from retailgr import privacy


@dataclass
class BundleManifest:
    """What produced this bundle, so a served response can be traced back."""

    model_version: str
    model_type: str
    variant: str
    dataset: str
    vocab_size: int
    embedding_dim: int
    max_len: int
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    model_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    # Present when a ranker was exported alongside the retrieval model. A
    # bundle without it serves retrieval order, which is a valid deployment.
    ranker_config: dict[str, Any] | None = None
    ranker_metrics: dict[str, Any] = field(default_factory=dict)
    # The fitted per-head calibrators. These live beside the weights rather
    # than in ``ranker_config`` on purpose: they are derived from held-out
    # data, not chosen by an operator, and a YAML file someone can edit into
    # disagreement with the model it corrects is the wrong home for them.
    head_calibration: dict[str, Any] | None = None
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    )


class ServingBundle:
    """A loaded bundle, ready to serve."""

    def __init__(
        self,
        manifest: BundleManifest,
        model: Any,
        item_embeddings: np.ndarray,
        token_by_id: dict[int, str],
        category_by_id: dict[int, str],
        skus_by_token: dict[str, list[str]],
        product_by_token: dict[str, str],
        ranker: Any | None = None,
    ):
        self.manifest = manifest
        self.model = model
        self.item_embeddings = item_embeddings
        self.token_by_id = token_by_id
        self.category_by_id = category_by_id
        self.skus_by_token = skus_by_token
        self.product_by_token = product_by_token
        self.ranker = ranker
        self.id_by_token = {token: token_id for token_id, token in token_by_id.items()}

    @property
    def has_ranker(self) -> bool:
        return self.ranker is not None

    @property
    def model_version(self) -> str:
        return self.manifest.model_version

    def skus_for(self, token: str) -> list[str]:
        """The purchasable SKUs behind a model token.

        A token is what the model ranks; a SKU is what the customer buys. The
        policy layer needs this mapping to check stock, and the response needs
        it to suggest something orderable.
        """
        return self.skus_by_token.get(token, [token])


# -- export -------------------------------------------------------------------


def export_bundle(
    output_dir: Path,
    model: Any,
    manifest: BundleManifest,
    token_by_id: dict[int, str],
    category_by_id: dict[int, str],
    skus_by_token: dict[str, list[str]],
    product_by_token: dict[str, str],
    ranker: Any | None = None,
) -> Path:
    """Write a bundle to ``output_dir``."""
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    net = model.net
    torch.save(net.state_dict(), output_dir / "model.pt")
    if ranker is not None:
        torch.save(ranker.net.state_dict(), output_dir / "ranker.pt")

    embeddings = net.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
    np.save(output_dir / "item_embeddings.npy", embeddings)

    # These two are item data and the scrub is a no-op on them. They go
    # through it anyway so the rule has no exceptions: an allowlist of
    # "files we decided are fine" is the structure that rots, because the
    # next entry is added by whoever is in a hurry.
    (output_dir / "vocab.json").write_text(
        json.dumps(
            privacy.scrub(
                {
                    "token_by_id": {str(k): v for k, v in token_by_id.items()},
                    "category_by_id": {str(k): v for k, v in category_by_id.items()},
                }
            )
        ),
        encoding="utf-8",
    )
    (output_dir / "hierarchy.json").write_text(
        json.dumps(
            privacy.scrub({"skus_by_token": skus_by_token, "product_by_token": product_by_token})
        ),
        encoding="utf-8",
    )
    # Scrubbed here and not at the call sites that build `ranker_metrics`.
    # `experiment.py` already strips `user_ids` by hand in two places; the
    # export path was the third and it was forgotten, which put 200 real
    # customer ids into this file and — because `/v1/model` returns the
    # manifest verbatim — onto an unauthenticated HTTP endpoint. Enforcing
    # it at the boundary is the difference between a rule and a habit.
    (output_dir / "manifest.json").write_text(
        json.dumps(privacy.scrub(asdict(manifest)), indent=2), encoding="utf-8"
    )
    return output_dir


def load_bundle(bundle_dir: Path, device: str = "cpu") -> ServingBundle:
    """Load a bundle written by :func:`export_bundle`."""
    import torch

    from retailgr.models.hstu import HSTUModel
    from retailgr.models.sasrec import SASRecModel

    bundle_dir = Path(bundle_dir)
    manifest_data = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = BundleManifest(**manifest_data)

    model_config = dict(manifest.model_config)
    model_config["device"] = device
    classes = {"sasrec": SASRecModel, "hstu": HSTUModel}
    if manifest.model_type not in classes:
        raise ValueError(f"bundle has unknown model_type '{manifest.model_type}'")
    model = classes[manifest.model_type](manifest.vocab_size, model_config)
    state = torch.load(bundle_dir / "model.pt", map_location=device, weights_only=True)
    model.net.load_state_dict(state)
    model.net.eval()

    embeddings = np.load(bundle_dir / "item_embeddings.npy")
    vocab = json.loads((bundle_dir / "vocab.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((bundle_dir / "hierarchy.json").read_text(encoding="utf-8"))

    ranker = None
    ranker_path = bundle_dir / "ranker.pt"
    if manifest.ranker_config and ranker_path.exists():
        from retailgr.models.ranker import HSTURanker

        ranker_config = dict(manifest.ranker_config)
        ranker_config["device"] = device
        ranker = HSTURanker(manifest.vocab_size, ranker_config)
        ranker.net.load_state_dict(
            torch.load(ranker_path, map_location=device, weights_only=True)
        )
        ranker.net.eval()
        if manifest.head_calibration:
            from retailgr.evaluation.calibration import HeadCalibrators

            # Without this the served blend is the uncalibrated one — the
            # weights would be applied to probabilities on incomparable
            # scales, which is the failure the gate now refuses to ship.
            ranker.calibrators = HeadCalibrators.from_dict(manifest.head_calibration)

    return ServingBundle(
        manifest=manifest,
        model=model,
        item_embeddings=embeddings,
        token_by_id={int(k): v for k, v in vocab["token_by_id"].items()},
        category_by_id={int(k): v for k, v in vocab["category_by_id"].items()},
        skus_by_token={k: list(v) for k, v in hierarchy["skus_by_token"].items()},
        product_by_token=dict(hierarchy["product_by_token"]),
        ranker=ranker,
    )
