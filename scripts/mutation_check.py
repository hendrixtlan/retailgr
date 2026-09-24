#!/usr/bin/env python
"""Break each load-bearing claim on purpose; check that something fails.

A green suite proves the tests pass. It does not prove they would notice if
the thing they describe stopped being true — and the difference is not
academic. Running this the first time found that the map-type test written to
guard a real pipeline defect covered a different code path from the defect:
reverting the fix left the test green.

Each entry below is a claim this project makes in its README and rests on in
production. The mutation is the plausible regression — usually the exact
state the code was in before the bug was found — and the target is the test
that is supposed to catch it.

    make mutants

Anything reported NOT CAUGHT is a claim with no guard behind it. The source
tree is restored whether or not the run succeeds.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent

# (name, file, before, after, pytest target, slow)
MUTATIONS: list[tuple[str, str, str, str, str, bool]] = [
    (
        "M-FALCON normaliser depends on the candidate count",
        "src/retailgr/models/ranker.py",
        "normaliser = float(self.max_len)",
        "normaliser = float(items.shape[1])",
        "tests/test_ranker.py -k mfalcon",
        False,
    ),
    (
        "the in-process broker partitions with crc32 again",
        "src/retailgr/streaming/broker.py",
        'return (murmur2(key.encode("utf-8")) & 0x7FFFFFFF) % partitions',
        'import zlib; return zlib.crc32(key.encode("utf-8")) % partitions',
        "tests/test_streaming.py -k partition",
        False,
    ),
    (
        "a calibrator may reverse a head's ranking",
        "src/retailgr/evaluation/calibration.py",
        "    if not _preserves_ordering(candidate, p):\n        return IdentityCalibrator()",
        "    if False:\n        return IdentityCalibrator()",
        "tests/test_calibration.py -k reverse",
        False,
    ),
    (
        "the offline gate passes when it cannot be evaluated",
        "src/retailgr/evaluation/ranking.py",
        '            "reason": "the gate could not be evaluated; ranker left out of the bundle",',
        '            "passed": True, "reason": "unmeasured",',
        "tests/test_ranker.py -k gate",
        False,
    ),
    (
        "FAISS probes a tenth of the cells again",
        "src/retailgr/serving/retrieval.py",
        "self.nprobe = int(nprobe) if nprobe else max(1, self.nlist // 2)",
        "self.nprobe = int(nprobe) if nprobe else max(1, self.nlist // 10)",
        "tests/test_backend_contracts.py -k approximate",
        False,
    ),
    (
        "Iceberg reaches Spark through pandas again",
        "src/retailgr/io/tables.py",
        "        return spark.createDataFrame(data)",
        "        return spark.createDataFrame(data.to_pandas())",
        "tests/test_iceberg.py -k arrow_reaches_spark",
        False,
    ),
    (
        "the paired test calls everything significant",
        "src/retailgr/evaluation/stats.py",
        "return self.ci_low > 0 or self.ci_high < 0",
        "return True",
        "tests/test_stats.py -k difference",
        False,
    ),
    (
        "the time split goes back to an approximate quantile",
        "src/retailgr/jobs/sequences.py",
        'error = float(cfg.get("sequences.quantile_error", 0.0))',
        'error = float(cfg.get("sequences.quantile_error", 0.001))',
        "tests/test_warehouse_equivalence.py -k cutoffs",
        True,
    ),
    # -- privacy. Every one of these is the state the code was actually in
    # before this pass, so "caught" means the guard would have fired then.
    (
        "the bundle manifest is written without scrubbing identifiers",
        "src/retailgr/serving/bundle.py",
        'json.dumps(privacy.scrub(asdict(manifest)), indent=2), encoding="utf-8"',
        'json.dumps(asdict(manifest), indent=2), encoding="utf-8"',
        "tests/test_privacy.py -k export_bundle",
        False,
    ),
    (
        "/v1/model returns the manifest verbatim",
        "src/retailgr/serving/api.py",
        "return privacy.scrub(asdict(service.bundle.manifest))",
        "return asdict(service.bundle.manifest)",
        "tests/test_privacy.py -k api_response",
        False,
    ),
    (
        "consent fails open instead of closed",
        "src/retailgr/privacy.py",
        "    if not value:\n        return False",
        "    if not value:\n        return True",
        "tests/test_privacy.py -k fails_closed",
        False,
    ),
    (
        "the erasure verifier greps Parquet instead of decoding it",
        "src/retailgr/erasure.py",
        '    if path.suffix == ".parquet":\n'
        "        return _count_in_parquet(path, identifiers)",
        '    if path.suffix == ".parquet" and False:\n'
        "        return _count_in_parquet(path, identifiers)",
        "tests/test_erasure.py -k compressed_parquet",
        False,
    ),
    (
        "erasure leaves the raw file that everything is re-derived from",
        "src/retailgr/erasure.py",
        "    report.steps.append(_erase_from_raw(cfg, identifiers, dry_run))",
        "    report.steps.append({'step': 'raw', 'status': 'skipped'})",
        "tests/test_erasure.py -k rewrites_the_raw_file",
        False,
    ),
    (
        "the Iceberg erasure skips snapshot expiry, leaving the rows readable",
        "src/retailgr/erasure.py",
        "        report.steps.append(_expire_snapshots(cfg, [t for t, _ in tables], dry_run))",
        "        report.steps.append({'step': 'expire_snapshots', 'status': 'expired'})",
        "tests/test_erasure.py -k expires_the_snapshots",
        False,
    ),
    (
        "the shared cold-start list moves back into the per-user keyspace",
        "src/retailgr/online_store.py",
        'return f"{self.namespace}:fallback_global"',
        'return f"{self.namespace}:fallback:__global__"',
        "tests/test_erasure.py -k outside_the_user_keyspace",
        False,
    ),
    (
        "serving tails lose their TTL and are kept until the disk fills",
        "src/retailgr/online_store.py",
        "        if self.ttl_seconds:\n            # Refreshed on every write",
        "        if False:\n            # Refreshed on every write",
        "tests/test_erasure.py -k carry_a_ttl",
        False,
    ),
    # -- failure modes. Each is the state the request path was actually in.
    (
        "a dead online store raises instead of degrading",
        "src/retailgr/serving/service.py",
        "            response = self._fallback_response(\n"
        '                request_id, context, limit, timings, "store_error"',
        "            raise  # noqa\n"
        '            response = self._fallback_response(\n'
        '                request_id, context, limit, timings, "store_error"',
        "tests/test_failure_modes.py -k dead_online_store",
        False,
    ),
    (
        "the fallback path goes back to needing the store it recovers from",
        "src/retailgr/serving/service.py",
        "            product_ids = list(self._last_resort)",
        "            raise",
        "tests/test_failure_modes.py -k does_not_require_the_store",
        False,
    ),
    (
        "a throwing ranker becomes a 500 again",
        "src/retailgr/serving/service.py",
        "        except Exception:\n"
        "            # Degrade to retrieval order rather than to the popular list.",
        "        except ZeroDivisionError:\n"
        "            # Degrade to retrieval order rather than to the popular list.",
        "tests/test_failure_modes.py -k throwing_ranker",
        False,
    ),
    (
        "an inventory read failure empties the page instead of failing open",
        "src/retailgr/serving/service.py",
        "            item_states = {}",
        "            raise",
        "tests/test_failure_modes.py -k inventory_read",
        False,
    ),
    # -- the ranker's history loss, which is per-head for a measured reason.
    (
        "a head left out of the history-loss map is silently switched off",
        "src/retailgr/models/ranker.py",
        "                head: float(weight.get(head, 1.0)) for head in HEADS",
        "                head: float(weight.get(head, 0.0)) for head in HEADS",
        "tests/test_slate.py -k unlisted_head",
        False,
    ),
    (
        "the retention guard stops refusing and empties the table",
        "src/retailgr/jobs/retention.py",
        "    if share > max_share:\n        raise RetentionRefused(",
        "    if False:\n        raise RetentionRefused(",
        "tests/test_retention.py -k refuses_on_the_sample_data",
        True,
    ),
    # -- early stopping. Each is a way it goes wrong without any error.
    (
        "early stopping leaves dropout off after the first validation check",
        "src/retailgr/models/training.py",
        "        net.train()\n\n        if self.epochs_without_improvement",
        "        pass\n\n        if self.epochs_without_improvement",
        "tests/test_early_stopping.py -k back_in_training_mode",
        False,
    ),
    (
        "the best weights are kept as a live reference, not a copy",
        "src/retailgr/models/training.py",
        "            self._best_state = copy.deepcopy(net.state_dict())",
        "            self._best_state = net.state_dict()",
        "tests/test_early_stopping.py -k copy_not_a_live_reference",
        False,
    ),
    (
        "early stopping returns the last weights instead of the best",
        "src/retailgr/models/training.py",
        "        if self._best_state is not None:\n"
        "            net.load_state_dict(self._best_state)",
        "        if False:\n            net.load_state_dict(self._best_state)",
        "tests/test_early_stopping.py -k best_checkpoint_not_the_current",
        False,
    ),
    (
        "the ablation harness selects epochs on the test split",
        "src/retailgr/experiment.py",
        "            fit_stats = model.fit(data.train, data.val)\n"
        "            evaluation = evaluate(",
        "            fit_stats = model.fit(data.train, data.test)\n"
        "            evaluation = evaluate(",
        "tests/test_early_stopping.py -k test_split_to_fit",
        False,
    ),
    (
        "the ablation report goes back to rendering only the deepest cutoff",
        "src/retailgr/experiment.py",
        '    cutoffs = sorted({min(result["k_values"]), k})',
        "    cutoffs = [k]",
        "tests/test_early_stopping.py -k top_ten_is_reported",
        False,
    ),
    (
        "exclude_seen goes back to one global answer for every surface",
        "src/retailgr/serving/service.py",
        "        if isinstance(self.exclude_seen, bool):\n            return self.exclude_seen",
        "        if True:\n            return bool(self.exclude_seen)",
        "tests/test_slate.py -k reminder_surfaces",
        False,
    ),
    (
        "the Spark consent filter stops trimming, denying 'service, analytics'",
        "src/retailgr/privacy.py",
        'F.split(F.coalesce(F.col(column), F.lit("")), ","), lambda part: F.trim(part)',
        'F.split(F.coalesce(F.col(column), F.lit("")), ","), F.trim',
        "tests/test_privacy_spark.py -k agrees_with_the_python_one",
        True,
    ),
]


def run(target: str, timeout: int) -> tuple[bool, str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *target.split(),
            "-q",
            "-x",
            "-p",
            "no:warnings",
            "--no-header",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=timeout,
    )
    summary = [
        line
        for line in result.stdout.strip().splitlines()
        if any(word in line for word in ("passed", "failed", "error"))
    ]
    return result.returncode != 0, (summary[-1] if summary else "no summary")[:70]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="also mutate claims whose guard needs the full Spark pipeline",
    )
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    selected = [m for m in MUTATIONS if args.include_slow or not m[5]]
    results: list[tuple[str, str, str]] = []

    for name, path, before, after, target, _slow in selected:
        source = REPO / path
        original = source.read_text(encoding="utf-8")
        if before not in original:
            # The code moved on and the mutation no longer applies. That is a
            # finding too: this entry is no longer guarding anything.
            results.append((name, "STALE — text not found", path))
            continue

        backup = tempfile.NamedTemporaryFile(delete=False, suffix=".bak").name
        shutil.copy(source, backup)
        try:
            source.write_text(original.replace(before, after, 1), encoding="utf-8")
            caught, detail = run(target, args.timeout)
            results.append((name, "caught" if caught else "NOT CAUGHT", detail))
        finally:
            shutil.copy(backup, source)
            pathlib.Path(backup).unlink(missing_ok=True)

    width = max(len(name) for name, _, _ in results) + 2
    print(f"\n{'claim broken on purpose':{width}s} {'result':22s} detail")
    print("-" * (width + 90))
    for name, verdict, detail in results:
        print(f"{name:{width}s} {verdict:22s} {detail}")

    missed = [r for r in results if r[1] != "caught"]
    print(f"\n{len(results) - len(missed)}/{len(results)} caught")
    if missed:
        print("\nA claim with no guard behind it:")
        for name, verdict, _ in missed:
            print(f"  - {name} ({verdict})")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
