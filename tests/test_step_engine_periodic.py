import unittest

import torch
from torch.nn import functional as F

from sample_fg.estimators import EMAEstimator, ExactEstimator, PeriodicEstimator
from sample_fg.full_gradient import FullGradientResult, FullGradientSweepMetadata
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import derive_auxiliary_seed
from sample_fg.step_engine import StepEngine


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.4, -0.2]))

    def forward(self, x):
        return x @ self.weight


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


def _batch(step=0):
    return (
        torch.tensor([[1.0 + 0.1 * step, 2.0], [-0.5, 1.5 - 0.1 * step]]),
        torch.tensor([0.3, -0.8]),
    )


def _loss(model, batch):
    return F.mse_loss(model(batch[0]), batch[1])


def _metadata(index, step, purpose):
    return FullGradientSweepMetadata(
        sample_count=4,
        micro_batch_count=2,
        configured_micro_batch_size=2,
        observed_micro_batch_sizes=(2, 2),
        forward_calls=2,
        autograd_grad_calls=2,
        mean_loss=1.0,
        elapsed_s=0.01,
        precision_mode="fp32",
        param_index_fingerprint=index.fingerprint,
        source_fingerprint="periodic-fixture",
        seed=derive_auxiliary_seed(
            protocol_seed=1,
            dataset="fixture",
            shots=2,
            config_hash="periodic-fixture-config",
            optimizer_step=step,
            purpose=purpose,
        ),
    )


class _SequenceService:
    def __init__(self, index, values):
        self.index = index
        self.values = tuple(tuple(value) for value in values)
        self.calls = []

    def compute(self, *, optimizer_step, purpose):
        self.calls.append((optimizer_step, purpose))
        value = self.values[optimizer_step % len(self.values)]
        state = GradientState.from_tensors(
            self.index,
            (torch.tensor(value, dtype=torch.float32, device=self.index[0].device),),
        )
        return FullGradientResult(
            gradient=state,
            metadata=_metadata(self.index, optimizer_step, purpose),
        )


def _engine(model):
    index = ParamIndex.from_model(model)
    optimizer = _CountingSGD(index.parameters, lr=0.02)
    engine = StepEngine(
        param_index=index,
        optimizer=optimizer,
        precision_controller=PrecisionController("fp32"),
        rho=0.05,
        alpha=0.0015,
    )
    return index, optimizer, engine


class SAMPLePeriodicStepTest(unittest.TestCase):
    def test_k2_refresh_boundary_hard_reset_age_and_recurrence(self):
        model = _TinyModel()
        index, optimizer, engine = _engine(model)
        service = _SequenceService(index, ((1.0, -0.5), (9.0, 9.0), (-2.0, 0.75)))
        estimator = PeriodicEstimator(
            index,
            ema_lambda=0.15,
            refresh_k_steps=2,
            full_gradient_service=service,
        )
        records = [
            engine.step_sample(
                _batch(step),
                lambda item, model=model: _loss(model, item),
                estimator,
            )
            for step in range(4)
        ]
        self.assertEqual([record.estimator_result.refreshed for record in records], [True, False, True, False])
        self.assertEqual([record.estimator_result.age_steps for record in records], [0, 1, 0, 1])
        self.assertEqual([record.estimator_result.last_refresh_step for record in records], [0, 0, 2, 2])
        self.assertEqual(service.calls, [(0, "periodic_refresh"), (2, "periodic_refresh")])
        self.assertEqual([record.estimator_result.exact_query_count for record in records], [1, 1, 2, 2])
        for step in (0, 2):
            self.assertTrue(
                torch.equal(
                    records[step].estimator_result.active_global_estimate[0],
                    records[step].estimator_result.exact_reference[0],
                )
            )
        expected1 = records[0].estimator_result.active_global_estimate.scale(0.15).add(records[1].batch_gradient.scale(0.85))
        expected3 = records[2].estimator_result.active_global_estimate.scale(0.15).add(records[3].batch_gradient.scale(0.85))
        torch.testing.assert_close(records[1].estimator_result.active_global_estimate.components, expected1.components)
        torch.testing.assert_close(records[3].estimator_result.active_global_estimate.components, expected3.components)
        self.assertEqual(optimizer.step_calls, 4)

    def test_k1_is_end_to_end_exact_alias_with_same_purpose_and_results(self):
        exact_model = _TinyModel()
        periodic_model = _TinyModel()
        exact_index, exact_optimizer, exact_engine = _engine(exact_model)
        periodic_index, periodic_optimizer, periodic_engine = _engine(periodic_model)
        values = ((1.25, -0.75),)
        exact_service = _SequenceService(exact_index, values)
        periodic_service = _SequenceService(periodic_index, values)
        exact = ExactEstimator(exact_index, full_gradient_service=exact_service)
        periodic = PeriodicEstimator(
            periodic_index,
            ema_lambda=0.15,
            refresh_k_steps=1,
            full_gradient_service=periodic_service,
        )
        batch = _batch()
        exact_record = exact_engine.step_sample(batch, lambda item: _loss(exact_model, item), exact)
        periodic_record = periodic_engine.step_sample(batch, lambda item: _loss(periodic_model, item), periodic)
        self.assertEqual(exact_service.calls, [(0, "optimization_exact")])
        self.assertEqual(periodic_service.calls, [(0, "optimization_exact")])
        for left, right in (
            (exact_record.estimator_result.active_global_estimate, periodic_record.estimator_result.active_global_estimate),
            (exact_record.projection.batch_component, periodic_record.projection.batch_component),
            (exact_record.sam_perturbation, periodic_record.sam_perturbation),
            (exact_record.total_displacement, periodic_record.total_displacement),
            (exact_record.perturbed_gradient, periodic_record.perturbed_gradient),
            (exact_record.final_gradient, periodic_record.final_gradient),
        ):
            torch.testing.assert_close(left.components, right.components)
        torch.testing.assert_close(exact_model.weight, periodic_model.weight)
        self.assertEqual(exact_optimizer.step_calls, periodic_optimizer.step_calls)

    def test_end_to_end_method_query_behavior_is_defining_distinction(self):
        query_counts = {}
        refreshes = {}
        for mode in ("ema", "exact", "periodic"):
            model = _TinyModel()
            index, optimizer, engine = _engine(model)
            service = _SequenceService(index, ((1.0, -0.5), (-2.0, 0.75), (0.5, 1.5)))
            if mode == "ema":
                estimator = EMAEstimator(index, ema_lambda=0.15)
            elif mode == "exact":
                estimator = ExactEstimator(index, full_gradient_service=service)
            else:
                estimator = PeriodicEstimator(
                    index,
                    ema_lambda=0.15,
                    refresh_k_steps=2,
                    full_gradient_service=service,
                )
            mode_records = [
                engine.step_sample(_batch(step), lambda item, model=model: _loss(model, item), estimator)
                for step in range(3)
            ]
            query_counts[mode] = estimator.exact_query_count
            refreshes[mode] = [record.estimator_result.refreshed for record in mode_records]
            self.assertEqual(optimizer.step_calls, 3)
        self.assertEqual(query_counts, {"ema": 0, "exact": 3, "periodic": 2})
        self.assertEqual(refreshes["periodic"], [True, False, True])


if __name__ == "__main__":
    unittest.main()
