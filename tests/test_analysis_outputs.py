from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from analysis.make_tables import AnalysisInputError, SMOKE_LABEL, format_stat, make_tables
from analysis.plot_diagnostics import make_plots
from tests.task24_fixture import write_fixture


class AnalysisOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.aggregate = write_fixture(self.root / "aggregate")

    def tearDown(self):
        self.temporary.cleanup()

    def test_required_table_inventory_mean_std_n_and_order(self):
        manifest = make_tables(self.aggregate, self.root / "tables")
        self.assertEqual(
            set(manifest["tables"]),
            {"reproduction_summary", "extension_summary", "few_shot_summary", "raw_seed_results", "efficiency_summary", "periodic_k_selection", "mechanism_summary", "paired_differences"},
        )
        text = (self.root / "tables" / "reproduction_summary.md").read_text(encoding="utf-8")
        self.assertIn("± 1.000 (n=3)", text)
        with (self.root / "tables" / "reproduction_summary.csv").open(encoding="utf-8") as stream:
            rows = list(__import__("csv").DictReader(stream))
        self.assertEqual([row["method"] for row in rows[:3]], ["CoOp", "SAM", "SAMPLe-EMA"])
        self.assertEqual(rows[0]["base_n"], "3")
        extension_header = (self.root / "tables" / "extension_summary.csv").read_text(encoding="utf-8").splitlines()[0]
        for field in (
            "estimator_exact_log_norm_ratio_mean_std",
            "batch_component_exact_abs_cosine_mean_std",
            "perturbed_gradient_exact_abs_cosine_mean_std",
            "overhead_vs_ema_pct_mean_std",
            "full_gradient_time_s_mean_std",
            "exact_sweeps_mean_std",
        ):
            self.assertIn(field, extension_header)
        k_header = (
            self.root / "tables" / "periodic_k_selection.csv"
        ).read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("estimator_exact_cosine_mean_std", k_header)
        self.assertIn("exact_sweeps_mean_std", k_header)
        self.assertNotIn("base_mean", k_header.lower())
        self.assertNotIn("new_mean", k_header.lower())
        self.assertNotIn("hm_mean", k_header.lower())

    def test_n1_has_no_fabricated_error_bar(self):
        self.assertEqual(format_stat(12.5, None, 1), "12.500 (n=1; std unavailable)")

    def test_paired_direction_and_no_significance_theater(self):
        manifest = make_tables(self.aggregate, self.root / "tables")
        paired = (self.root / "tables" / "paired_differences.md").read_text(encoding="utf-8")
        self.assertIn("candidate_minus_baseline", paired)
        self.assertFalse(manifest["statistics"]["significance_stars"])

    def test_scientific_default_rejects_smoke_and_explicit_mode_labels_it(self):
        smoke = self.root / "smoke"
        shutil.copytree(self.aggregate, smoke)
        report = json.loads((smoke / "aggregation_report.json").read_text())
        report.update({"mode": "smoke", "scientific_default": False, "scientific_rows": 0, "smoke_rows": report["eligible_runs"]})
        (smoke / "aggregation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        runs_path = smoke / "runs_long.json"
        runs = json.loads(runs_path.read_text())
        for row in runs["rows"]:
            row["smoke"] = True
        runs_path.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(AnalysisInputError, "does not match"):
            make_tables(smoke, self.root / "bad")
        table_manifest = make_tables(smoke, self.root / "smoke-tables", mode="smoke")
        plot_manifest = make_plots(smoke, self.root / "smoke-plots", mode="smoke")
        self.assertEqual(table_manifest["display_label"], SMOKE_LABEL)
        self.assertEqual(plot_manifest["display_label"], SMOKE_LABEL)
        self.assertIn(SMOKE_LABEL, (self.root / "smoke-tables" / "reproduction_summary.md").read_text(encoding="utf-8"))
        self.assertIn("mode=smoke", (self.root / "smoke-plots" / "global_estimate_fidelity.svg").read_text(encoding="utf-8"))

    def test_plot_inventory_axes_raw_points_and_distinct_orthogonality(self):
        manifest = make_plots(self.aggregate, self.root / "plots")
        expected = {
            "global_estimate_fidelity", "estimator_relative_l2", "estimator_norm_ratio",
            "batch_component_estimator_alignment", "construction_orthogonality",
            "reference_construction_orthogonality", "perturbed_estimator_alignment",
            "objective_orthogonality", "perturbed_batch_component_alignment",
            "perturbed_batch_alignment", "taylor_exploitation", "taylor_exploration",
            "taylor_joint",
            "base_new_hm_by_estimator", "shot_count_base", "shot_count_new", "shot_count_hm",
            "hm_vs_estimator_error", "runtime_overhead", "periodic_k_fidelity",
            "periodic_k_efficiency", "periodic_refresh_age",
        }
        self.assertEqual(set(manifest["plots"]), expected)
        self.assertFalse(manifest["orthogonality_concepts"]["collapsed"])
        self.assertEqual(manifest["plots"]["construction_orthogonality"]["metric_keys"], ["grad/batch_component_exact_cosine"])
        self.assertEqual(manifest["plots"]["objective_orthogonality"]["metric_keys"], ["grad/perturbed_gradient_exact_cosine"])
        self.assertTrue(manifest["plots"]["base_new_hm_by_estimator"]["raw_seed_points"])
        self.assertIn("accuracy", manifest["plots"]["base_new_hm_by_estimator"]["y_axis"])

    def test_periodic_refresh_markers_and_sample_std_errorbars_are_encoded(self):
        make_plots(self.aggregate, self.root / "plots")
        refresh = (self.root / "plots" / "periodic_refresh_age.svg").read_text(encoding="utf-8")
        accuracy = (self.root / "plots" / "base_new_hm_by_estimator.svg").read_text(encoding="utf-8")
        self.assertIn('class="refresh-marker"', refresh)
        self.assertIn('class="sample-std-errorbar"', accuracy)

    def test_png_and_svg_are_structurally_valid(self):
        make_plots(self.aggregate, self.root / "plots")
        with Image.open(self.root / "plots" / "objective_orthogonality.png") as image:
            self.assertEqual(image.size, (1000, 640))
            self.assertEqual(image.format, "PNG")
        root = ET.parse(self.root / "plots" / "objective_orthogonality.svg").getroot()
        self.assertTrue(root.tag.endswith("svg"))

    def test_rendering_is_byte_deterministic(self):
        first, second = self.root / "first", self.root / "second"
        make_tables(self.aggregate, first / "tables"); make_plots(self.aggregate, first / "plots")
        make_tables(self.aggregate, second / "tables"); make_plots(self.aggregate, second / "plots")
        names = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
        self.assertGreater(len(names), 20)
        for name in names:
            a = hashlib.sha256((first / name).read_bytes()).digest()
            b = hashlib.sha256((second / name).read_bytes()).digest()
            self.assertEqual(a, b, str(name))

    def test_missing_required_aggregate_fails_clearly(self):
        (self.aggregate / "diagnostics_long.json").unlink()
        with self.assertRaisesRegex(AnalysisInputError, "diagnostics_long.json"):
            make_plots(self.aggregate, self.root / "plots")


if __name__ == "__main__":
    unittest.main()
