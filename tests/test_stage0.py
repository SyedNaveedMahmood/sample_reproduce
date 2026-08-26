from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from sample_fg.results import RunArtifactError
from scripts.smoke_stage0_local import REQUIRED_ARTIFACTS, audit_run_directory


class Stage0ArtifactAuditTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        run = root / "run"
        (run / "checkpoints").mkdir(parents=True)
        (run / "logs").mkdir()
        (run / "config.yaml").write_text(
            yaml.safe_dump({"run": {"smoke": True}, "smoke": {"allow_scientific_summary": False}}),
            encoding="utf-8",
        )
        (run / "environment.json").write_text(json.dumps({"gpu": {"name": "fixture"}}), encoding="utf-8")
        (run / "data_manifest.json").write_text(json.dumps({"selected_samples": ["x"]}), encoding="utf-8")
        common = {"schema_version": "sample_fg.metrics.v1", "event_type": "train_step", "run_id": "r"}
        (run / "metrics.jsonl").write_text(json.dumps(common) + "\n", encoding="utf-8")
        (run / "gradient_diagnostics.jsonl").write_text("", encoding="utf-8")
        (run / "summary.json").write_text(
            json.dumps({"status": "completed", "smoke": True, "allow_scientific_summary": False, "run_identity": {"run_id": "r"}}),
            encoding="utf-8",
        )
        (run / "checkpoints" / "final.pt").write_bytes(b"checkpoint")
        (run / "logs" / "run.log").write_text("completed PASS\n", encoding="utf-8")
        return run

    def test_complete_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = audit_run_directory(self._fixture(Path(temporary)))
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["required_artifacts"], len(REQUIRED_ARTIFACTS))

    def test_missing_artifact_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._fixture(Path(temporary))
            (run / "summary.json").unlink()
            with self.assertRaises(RunArtifactError):
                audit_run_directory(run)

    def test_failure_token_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._fixture(Path(temporary))
            (run / "logs" / "run.log").write_text("Traceback occurred\n", encoding="utf-8")
            with self.assertRaises(RunArtifactError):
                audit_run_directory(run)

    def test_scientific_smoke_gate_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = self._fixture(Path(temporary))
            (run / "summary.json").write_text(
                json.dumps({"status": "completed", "smoke": True, "allow_scientific_summary": True, "run_identity": {"run_id": "r"}}),
                encoding="utf-8",
            )
            with self.assertRaises(RunArtifactError):
                audit_run_directory(run)


if __name__ == "__main__":
    unittest.main()
