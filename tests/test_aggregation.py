from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sample_fg.environment import ENVIRONMENT_SCHEMA_VERSION
from sample_fg.results import METRICS_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION, atomic_write_json, atomic_write_yaml, resolve_config
from analysis.aggregate_results import AggregationError, aggregate


class AggregationTests(unittest.TestCase):
    def _run(
        self, root: Path, *, run_id: str, seed: int, method: str = "coop",
        estimator: str = "none", base: float = 70.0, new: float = 50.0,
        smoke: bool = False, allow: bool = True, lr: float = 0.002,
        diagnostic_step: int | None = None,
        periodic_k: int = 2,
        experiment_id: str = "X",
        dataset: str = "dtd",
    ) -> Path:
        run = root / run_id
        run.mkdir(parents=True)
        hm = 0.0 if base + new == 0 else 2 * base * new / (base + new)
        config = resolve_config(
            {
                "run": {"experiment_id": experiment_id, "run_id": run_id, "smoke": smoke},
                "data": {
                    "dataset": dataset, "shots": 16, "seed": seed,
                    "split_policy": "official_coop_fixed", "train_batch_size": 32,
                    "test_batch_size": 100, "num_workers": 8,
                    "preserve_upstream_drop_last": True,
                    "augmentation_policy": "pinned", "selected_source_fingerprint": "source",
                    "selected_count": 384 if dataset == "dtd" else 80,
                },
                "model": {
                    "backbone": "ViT-B/16", "prompt_learner": "CoOp", "effective_n_ctx": 4,
                    "ctx_init": "a photo of a", "class_specific_context": False,
                    "class_token_position": "end", "freeze_clip": True,
                    "checkpoint_sha256": "clip",
                },
                "method": {"name": method, "rho": 0.05 if method != "coop" else None, "alpha": 0.0015 if method == "sample" else None, "ema_lambda": 0.15 if estimator in {"ema", "periodic"} else None},
                "estimator": {"mode": estimator, "refresh_k_steps": periodic_k if estimator == "periodic" else None},
                "optim": {"name": "sgd", "lr": lr, "weight_decay": 0.0005, "momentum": 0.9, "nesterov": False, "max_epoch": 200, "scheduler": "cosine", "warmup_epoch": 1, "warmup_type": "constant", "warmup_cons_lr": 1e-5, "scheduler_step_unit": "epoch"},
                "runtime": {"precision": "coop_fp16", "gradient_state_dtype": "fp32"},
                "smoke": {"allow_scientific_summary": allow},
            }
        )
        config["run"]["run_id"] = run_id
        atomic_write_yaml(run / "config.yaml", config)
        atomic_write_json(run / "environment.json", {"schema_version": ENVIRONMENT_SCHEMA_VERSION, "gpu": {"name": "fixture"}})
        atomic_write_json(run / "data_manifest.json", {"schema_version": "sample_fg.data_manifest.v1", "official_split": {"sha256": "split"}, "complete_selected_source": {"count": 384 if dataset == "dtd" else 80}})
        (run / "metrics.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": METRICS_SCHEMA_VERSION,
                    "event_type": "train_step",
                    "run_id": run_id,
                    "optimizer_step": diagnostic_step,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        diagnostics = ""
        if diagnostic_step is not None:
            diagnostics = json.dumps({"schema_version": "sample_fg.diagnostic_event.v1", "optimizer_step": diagnostic_step, "metrics": {"grad/global_estimate_exact_cosine": 0.5}}) + "\n"
        (run / "gradient_diagnostics.jsonl").write_text(diagnostics, encoding="utf-8")
        atomic_write_json(
            run / "summary.json",
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "run_identity": {"run_id": run_id, "experiment_id": experiment_id},
                "status": "completed", "smoke": smoke,
                "allow_scientific_summary": allow,
                "evaluation": {"base_accuracy_pct": base, "new_accuracy_pct": new, "hm_pct": hm},
                "efficiency": {"train_total_s": float(seed), "full_gradient_total_s": 0.5, "peak_cuda_allocated_bytes": 10, "peak_cuda_reserved_bytes": 20, "compute_counts": {"optimizer_steps": 2, "full_gradient_sweeps": 1}},
                "estimator_diagnostics": {
                    "num_exact_reference_points": 1 if diagnostic_step is not None else 0,
                    "global_estimate_exact_cosine_mean": 0.5 if diagnostic_step is not None else None,
                    "global_estimate_exact_relative_l2_mean": 0.25 if diagnostic_step is not None else None,
                    "global_estimate_exact_log_norm_ratio_mean": 0.1 if diagnostic_step is not None else None,
                    "batch_component_exact_abs_cosine_mean": 0.02 if diagnostic_step is not None else None,
                    "perturbed_gradient_exact_abs_cosine_mean": 0.03 if diagnostic_step is not None else None,
                },
            },
        )
        return run

    def test_three_seed_mean_sample_std_and_n(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed, value in ((1, 60.0), (2, 70.0), (3, 80.0)):
                self._run(root, run_id=f"r{seed}", seed=seed, base=value, new=50.0)
            report = aggregate(root, root / "out")
            summary = json.loads((root / "out" / "summary_by_cell.json").read_text())["rows"][0]
            self.assertEqual(report["eligible_runs"], 3)
            self.assertEqual(summary["base_mean"], 70.0)
            self.assertEqual(summary["base_std"], 10.0)
            self.assertEqual(summary["base_n"], 3)

    def test_n1_std_is_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self._run(root, run_id="r", seed=1)
            aggregate(root, root / "out")
            row = json.loads((root / "out" / "summary_by_cell.json").read_text())["rows"][0]
            self.assertIsNone(row["hm_std"])

    def test_paired_differences_use_matching_seeds_and_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (1, 2, 3):
                self._run(root, run_id=f"c{seed}", seed=seed, base=60 + seed)
            for seed in (1, 2):
                self._run(root, run_id=f"e{seed}", seed=seed, method="sample", estimator="ema", base=65 + seed)
            aggregate(root, root / "out")
            row = json.loads((root / "out" / "paired_differences.json").read_text())["rows"][0]
            self.assertEqual(row["direction"], "candidate_minus_baseline")
            self.assertEqual(row["base_delta_mean"], 5.0)
            self.assertEqual(row["base_paired_n"], 2)

    def test_smoke_excluded_by_default_and_included_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self._run(root, run_id="s", seed=1, smoke=True, allow=False)
            scientific = aggregate(root, root / "science")
            smoke = aggregate(root, root / "smoke", mode="smoke")
            self.assertEqual(scientific["eligible_runs"], 0)
            self.assertEqual(smoke["eligible_runs"], 1)

    def test_duplicate_same_config_selects_earliest_not_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._run(root, run_id="a", seed=1, base=60)
            second = self._run(root, run_id="b", seed=1, base=99)
            # A true rerun has the same hash; only runtime/evaluation can differ.
            config = yaml_safe(first / "config.yaml")
            config["run"]["run_id"] = "b"
            atomic_write_yaml(second / "config.yaml", config)
            summary = json.loads((second / "summary.json").read_text())
            summary["run_identity"]["run_id"] = "b"
            atomic_write_json(second / "summary.json", summary)
            report = aggregate(root, root / "out")
            row = json.loads((root / "out" / "runs_long.json").read_text())["rows"][0]
            self.assertEqual(row["run_id"], "a")
            self.assertEqual(row["base_accuracy_pct"], 60.0)
            self.assertEqual(len(report["duplicate_attempts"]), 1)

    def test_conflicting_same_cell_config_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self._run(root, run_id="a", seed=1); self._run(root, run_id="b", seed=1, lr=0.01)
            with self.assertRaises(AggregationError): aggregate(root, root / "out")

    def test_incompatible_cross_method_protocol_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self._run(root, run_id="a", seed=1)
            second = self._run(root, run_id="b", seed=1, method="sample", estimator="ema")
            config = yaml_safe(second / "config.yaml"); config["data"]["train_batch_size"] = 7
            config = resolve_config(config); config["run"]["run_id"] = "b"; atomic_write_yaml(second / "config.yaml", config)
            with self.assertRaises(AggregationError): aggregate(root, root / "out")

    def test_malformed_jsonl_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = self._run(root, run_id="r", seed=1)
            (run / "metrics.jsonl").write_text("{bad\n", encoding="utf-8")
            with self.assertRaises(AggregationError): aggregate(root, root / "out")

    def test_hm_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = self._run(root, run_id="r", seed=1)
            summary = json.loads((run / "summary.json").read_text()); summary["evaluation"]["hm_pct"] = 99
            atomic_write_json(run / "summary.json", summary)
            with self.assertRaises(AggregationError): aggregate(root, root / "out")

    def test_diagnostics_efficiency_and_order_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run(root, run_id="sam", seed=1, method="sam")
            self._run(root, run_id="coop", seed=1, diagnostic_step=2)
            aggregate(root, root / "out")
            runs = json.loads((root / "out" / "runs_long.json").read_text())["rows"]
            diagnostics = json.loads((root / "out" / "diagnostics_long.json").read_text())["rows"]
            efficiency = json.loads((root / "out" / "efficiency.json").read_text())["rows"]
            self.assertEqual([row["method_key"] for row in runs], ["coop:none", "sam:none"])
            self.assertEqual(diagnostics[0]["optimizer_step"], 2)
            self.assertEqual(efficiency[0]["optimizer_steps"], 2)

    def test_periodic_age_and_matched_hardware_overhead_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ema = self._run(root, run_id="ema", seed=1, method="sample", estimator="ema", diagnostic_step=0)
            periodic = self._run(root, run_id="periodic", seed=1, method="sample", estimator="periodic", diagnostic_step=0)
            metrics = json.loads((periodic / "metrics.jsonl").read_text())
            metrics["estimator/age_steps"] = 0
            (periodic / "metrics.jsonl").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
            aggregate(root, root / "out")
            diagnostics = json.loads((root / "out" / "diagnostics_long.json").read_text())["rows"]
            efficiency = json.loads((root / "out" / "efficiency.json").read_text())["rows"]
            periodic_diag = next(row for row in diagnostics if row["estimator_mode"] == "periodic")
            periodic_eff = next(row for row in efficiency if row["estimator_mode"] == "periodic")
            periodic_summary = next(
                row
                for row in json.loads((root / "out" / "summary_by_cell.json").read_text())["rows"]
                if row["estimator_mode"] == "periodic"
            )
            self.assertEqual(periodic_diag["estimator_age_steps"], 0)
            self.assertEqual(periodic_eff["overhead_pair_status"], "matched_same_hardware")
            self.assertEqual(periodic_eff["train_time_overhead_vs_sample_ema_pct"], 0.0)
            self.assertEqual(periodic_summary["estimator_exact_log_norm_ratio_mean"], 0.1)
            self.assertEqual(periodic_summary["batch_component_exact_abs_cosine_mean"], 0.02)
            self.assertEqual(periodic_summary["perturbed_gradient_exact_abs_cosine_mean"], 0.03)

    def test_periodic_k_values_are_distinct_planned_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._run(
                root, run_id="k2", seed=1, method="sample",
                estimator="periodic", periodic_k=2,
            )
            self._run(
                root, run_id="k4", seed=1, method="sample",
                estimator="periodic", periodic_k=4,
            )
            aggregate(root, root / "out")
            rows = json.loads(
                (root / "out" / "summary_by_cell.json").read_text()
            )["rows"]
            self.assertEqual(
                [row["periodic_k_steps"] for row in rows], [2, 4]
            )
            self.assertEqual(
                [row["method_key"] for row in rows],
                ["sample:periodic_k2", "sample:periodic_k4"],
            )

    def test_e2_opt_in_reuses_all_six_immutable_r2_ema_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset in ("dtd", "eurosat"):
                for seed in (1, 2, 3):
                    self._run(
                        root,
                        run_id=f"r2-{dataset}-{seed}",
                        dataset=dataset,
                        seed=seed,
                        method="sample",
                        estimator="ema",
                        experiment_id="R2",
                    )
            report = aggregate(
                root,
                root / "out",
                baseline_method_key="sample:ema",
                reuse_r2_ema_for_e2=True,
            )
            self.assertEqual(len(report["reused_artifacts"]), 6)
            runs = json.loads((root / "out" / "runs_long.json").read_text())["rows"]
            aliases = [row for row in runs if row["experiment_id"] == "E2"]
            self.assertEqual(len(aliases), 6)
            self.assertTrue(all(row["artifact_reused"] for row in aliases))
            self.assertEqual(
                {row["reuse_source_experiment_id"] for row in aliases}, {"R2"}
            )


def yaml_safe(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
