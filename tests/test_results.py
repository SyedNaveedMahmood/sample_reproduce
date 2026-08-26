"""Task-20 run-artifact and environment schema tests."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from sample_fg.environment import ENVIRONMENT_SCHEMA_VERSION, capture_environment
from sample_fg.results import (
    METRICS_SCHEMA_VERSION,
    RESOLVED_CONFIG_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    RunAccounting,
    RunArtifactError,
    RunArtifacts,
    RunIdentity,
    atomic_write_json,
    bind_run_identity,
    load_jsonl,
    resolve_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return {
        "run": {"experiment_id": "T20", "smoke": True},
        "data": {"dataset": "dtd", "shots": 4, "seed": 1},
        "method": {"name": "sample", "rho": 0.05, "alpha": 0.0015},
        "estimator": {"mode": "periodic", "refresh_k_steps": 2},
        "runtime": {"precision": "fp32"},
        "smoke": {"allow_scientific_summary": False},
    }


def _identity(config: dict[str, object]) -> RunIdentity:
    return RunIdentity(
        dataset="dtd",
        shots=4,
        method_tag="sample",
        estimator_tag="periodic_k2",
        seed=1,
        utc_timestamp="20260815T120000000000Z",
        config_sha256=str(config["config_sha256"]),
        experiment_id="T20",
        smoke=True,
        allow_scientific_summary=False,
    )


class RunArtifactTests(unittest.TestCase):
    def test_resolved_config_hash_is_deterministic_and_sensitive(self):
        first = resolve_config(_config())
        second = resolve_config(dict(reversed(list(_config().items()))))
        self.assertEqual(first["schema_version"], RESOLVED_CONFIG_SCHEMA_VERSION)
        self.assertEqual(first["config_sha256"], second["config_sha256"])
        changed = _config()
        changed["data"] = {"dataset": "dtd", "shots": 8, "seed": 1}
        self.assertNotEqual(first["config_sha256"], resolve_config(changed)["config_sha256"])

    def test_canonical_tree_and_manifest_are_complete_and_byte_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "source_manifest.json"
            manifest_bytes = b'{"schema_version":"fixture","samples":["x"]}\n'
            manifest.write_bytes(manifest_bytes)
            config = resolve_config(_config())
            identity = _identity(config)
            config = bind_run_identity(config, identity)
            artifacts = RunArtifacts(root / "runs", identity)
            run_dir = artifacts.create(
                resolved_config=config,
                environment={"schema_version": ENVIRONMENT_SCHEMA_VERSION},
                data_manifest_source=manifest,
            )
            self.assertEqual(
                run_dir.relative_to(root / "runs").as_posix(),
                f"dtd/shots_4/sample/periodic_k2/seed_1/20260815T120000000000Z_{str(config['config_sha256'])[:8]}",
            )
            for relative in (
                "config.yaml",
                "environment.json",
                "data_manifest.json",
                "metrics.jsonl",
                "gradient_diagnostics.jsonl",
                "logs/run.log",
            ):
                self.assertTrue((run_dir / relative).is_file(), relative)
            self.assertTrue((run_dir / "checkpoints").is_dir())
            self.assertEqual((run_dir / "data_manifest.json").read_bytes(), manifest_bytes)
            with (run_dir / "config.yaml").open(encoding="utf-8") as stream:
                self.assertEqual(yaml.safe_load(stream), config)
            with self.assertRaises(RunArtifactError):
                artifacts.create(
                    resolved_config=config,
                    environment={},
                    data_manifest_source=manifest,
                )

    def test_strict_jsonl_append_and_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            config = resolve_config(_config())
            identity = _identity(config)
            config = bind_run_identity(config, identity)
            artifacts = RunArtifacts(root / "runs", identity)
            artifacts.create(
                resolved_config=config,
                environment={"schema_version": ENVIRONMENT_SCHEMA_VERSION},
                data_manifest_source=manifest,
            )
            metric = {
                "schema_version": METRICS_SCHEMA_VERSION,
                "event_type": "train_step",
                "run_id": identity.run_id,
                "optimizer_step": 0,
                "loss/current": 1.25,
            }
            artifacts.append_metric(metric)
            artifacts.append_diagnostic(
                {
                    "schema_version": "sample_fg.diagnostic_event.v1",
                    "run_id": identity.run_id,
                    "optimizer_step": 0,
                    "metrics": {"grad/xi": 0.5},
                }
            )
            self.assertEqual(load_jsonl(artifacts.metrics_path), [metric])
            self.assertEqual(len(load_jsonl(artifacts.diagnostics_path)), 1)
            bad = dict(metric)
            bad["loss/current"] = math.nan
            with self.assertRaises(RunArtifactError):
                artifacts.append_metric(bad)
            self.assertEqual(load_jsonl(artifacts.metrics_path), [metric])

    def test_smoke_gate_and_summary_schema(self):
        config = resolve_config(_config())
        with self.assertRaises(RunArtifactError):
            RunIdentity(
                dataset="dtd", shots=4, method_tag="sample", estimator_tag="ema",
                seed=1, utc_timestamp="x", config_sha256=str(config["config_sha256"]),
                experiment_id="T20", smoke=True, allow_scientific_summary=True,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            identity = _identity(config)
            config = bind_run_identity(config, identity)
            artifacts = RunArtifacts(root / "runs", identity)
            artifacts.create(resolved_config=config, environment={}, data_manifest_source=manifest)
            summary = {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "run_identity": identity.as_dict(),
                "status": "completed",
                "allow_scientific_summary": False,
                "evaluation": {"base_accuracy_pct": None, "new_accuracy_pct": None, "hm_pct": None},
                "efficiency": RunAccounting().as_dict(),
                "estimator_diagnostics": {"num_exact_reference_points": 0},
                "artifacts": {},
            }
            path = artifacts.write_summary(summary)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), summary)
            rejected = dict(summary)
            rejected["allow_scientific_summary"] = True
            with self.assertRaises(RunArtifactError):
                artifacts.write_summary(rejected)

    def test_atomic_json_preserves_existing_target_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "state.json"
            target.write_text('{"old":true}\n', encoding="utf-8")
            with mock.patch("sample_fg.results.os.replace", side_effect=OSError("fixture")):
                with self.assertRaises(OSError):
                    atomic_write_json(target, {"new": True})
            self.assertEqual(target.read_text(encoding="utf-8"), '{"old":true}\n')
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_accounting_uses_explicit_timing_memory_and_counts(self):
        accounting = RunAccounting(
            train_total_s=2.5,
            full_gradient_total_s=0.75,
            peak_cuda_allocated_bytes=1024,
            peak_cuda_reserved_bytes=2048,
        )
        accounting.increment("forward_calls", 2)
        accounting.increment("optimizer_steps")
        self.assertEqual(accounting.compute_counts, {"forward_calls": 2, "optimizer_steps": 1})
        with self.assertRaises(RunArtifactError):
            accounting.increment("optimizer_steps", -1)

    def test_environment_contains_required_provenance(self):
        payload = capture_environment(
            project_repo=REPO_ROOT,
            coop_upstream_commit="ff61507c790454bce7c5052c3ac39e60772f1f89",
            dassl_commit="c61a1b570ac6333bd50fb5ae06aea59002fb20bb",
            precision_mode="fp32",
            clip_backbone="ViT-B/16",
            clip_checkpoint_identifier="ViT-B-16.pt",
            clip_checkpoint_sha256="5806e77c",
            capture_package_freeze=False,
        )
        self.assertEqual(payload["schema_version"], ENVIRONMENT_SCHEMA_VERSION)
        for group in ("git", "upstream", "python", "platform", "packages", "cuda", "cudnn", "gpu", "precision", "clip", "package_freeze"):
            self.assertIn(group, payload)
        self.assertRegex(payload["git"]["project_commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(payload["git"]["diff_hash"], r"^[0-9a-f]{64}$")

    def test_unsupported_objects_are_rejected_not_stringified(self):
        with self.assertRaises(RunArtifactError):
            resolve_config({"bad": Path("machine-specific")})


if __name__ == "__main__":
    unittest.main()
