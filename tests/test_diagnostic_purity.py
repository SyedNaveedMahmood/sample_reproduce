import copy
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from sample_fg.full_gradient import FullGradientService
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import RNGSnapshot, capture_rng_state, restore_rng_state


class _PurityDataset(Dataset):
    def __init__(self, *, fail_at=None, stochastic=True):
        self.inputs = torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.5],
                [0.5, 0.5, 1.0],
                [1.5, -0.5, 0.25],
                [-0.25, 0.75, 1.25],
                [2.0, 0.5, -1.0],
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)
        self.fail_at = fail_at
        self.stochastic = stochastic

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if index == self.fail_at:
            raise RuntimeError("intentional purity fixture failure")
        image = self.inputs[index].clone()
        if self.stochastic:
            noise = random.random() + float(np.random.random())
            noise += float(torch.rand(()).item())
            image = image + 0.01 * noise
        return {
            "img": image,
            "label": self.labels[index].clone(),
            "sample_id": f"purity/{index}.dat",
        }


class _PurityModel(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[0.25, -0.35], [0.5, 0.1], [-0.2, 0.4]],
                dtype=torch.float32,
                device=device,
            )
        )
        self.bias = nn.Parameter(torch.tensor([0.05, -0.05], device=device))
        self.frozen = nn.Parameter(
            torch.tensor([0.01, -0.01], device=device), requires_grad=False
        )
        self.dropout = nn.Dropout(p=0.2)
        self.register_buffer("static_buffer", torch.tensor([3.0], device=device))
        self._sample_fg_perturbation_active = False

    def forward(self, image):
        return self.dropout(image) @ self.weight + self.bias + self.frozen


def _loader(dataset, batch_size=2):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(999)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def _service(model, dataset=None, controller=None):
    loader = _loader(dataset or _PurityDataset())
    service = FullGradientService(
        model=model,
        param_index=ParamIndex.from_model(model),
        loader=loader,
        precision_controller=controller or PrecisionController("fp32"),
        protocol_seed=3,
        dataset="purity_fixture",
        shots=2,
        config_hash="task11-purity-v1",
    )
    return service, loader


def _rng_equal(left: RNGSnapshot, right: RNGSnapshot):
    return (
        left.python_state == right.python_state
        and left.numpy_state[0] == right.numpy_state[0]
        and np.array_equal(left.numpy_state[1], right.numpy_state[1])
        and left.numpy_state[2:] == right.numpy_state[2:]
        and torch.equal(left.torch_cpu_state, right.torch_cpu_state)
        and left.cuda_was_initialized == right.cuda_was_initialized
        and len(left.torch_cuda_states) == len(right.torch_cuda_states)
        and all(
            torch.equal(a, b)
            for a, b in zip(left.torch_cuda_states, right.torch_cuda_states)
        )
        and len(left.explicit_generators) == len(right.explicit_generators)
        and all(
            a.generator is b.generator
            and a.device == b.device
            and torch.equal(a.state, b.state)
            for a, b in zip(left.explicit_generators, right.explicit_generators)
        )
    )


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _batch_loss_and_gradient(model, batch):
    device = next(model.parameters()).device
    loss = F.cross_entropy(model(batch["img"].to(device)), batch["label"].to(device))
    gradient = torch.autograd.grad(
        loss,
        tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
        create_graph=False,
        retain_graph=False,
    )
    return float(loss.detach().item()), tuple(item.detach().clone() for item in gradient)


class DiagnosticPurityTest(unittest.TestCase):
    def setUp(self):
        random.seed(81)
        np.random.seed(81)
        torch.manual_seed(81)

    def test_none_tensor_and_mixed_grad_buffers_are_preserved_exactly(self):
        for case in ("none", "tensor", "mixed"):
            model = _PurityModel()
            if case == "none":
                model.weight.grad = None
                model.bias.grad = None
            elif case == "tensor":
                model.weight.grad = torch.full_like(model.weight, 4.0)
                model.bias.grad = torch.full_like(model.bias, -2.0)
            else:
                model.weight.grad = torch.full_like(model.weight, 4.0)
                model.bias.grad = None
            before = {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
            }
            service, _ = _service(model)
            service.compute(optimizer_step=0, purpose="diagnostic")
            for name, parameter in model.named_parameters():
                expected = before[name]
                if expected is None:
                    self.assertIsNone(parameter.grad, f"{case}:{name}")
                else:
                    self.assertTrue(torch.equal(expected, parameter.grad), f"{case}:{name}")

    def test_all_external_training_state_is_unchanged(self):
        model = _PurityModel()
        model.train()
        model.dropout.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        # Populate optimizer momentum before the purity snapshot.
        seed_loss = F.cross_entropy(
            model(torch.ones(2, 3)), torch.tensor([0, 1])
        )
        seed_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        controller = PrecisionController("fp32")
        service, loader = _service(model, controller=controller)
        model.weight.grad = torch.full_like(model.weight, 0.25)
        model.bias.grad = None

        parameters = {name: item.detach().clone() for name, item in model.named_parameters()}
        buffers = {name: item.detach().clone() for name, item in model.named_buffers()}
        grads = {
            name: None if item.grad is None else item.grad.detach().clone()
            for name, item in model.named_parameters()
        }
        modes = tuple(module.training for module in model.modules())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        precision_state = copy.deepcopy(controller.state_dict())
        precision_phase = controller.phase
        rng = capture_rng_state((loader.generator,))
        perturbation_active = model._sample_fg_perturbation_active

        service.compute(optimizer_step=7, purpose="diagnostic")

        self.assertTrue(all(torch.equal(parameters[n], p) for n, p in model.named_parameters()))
        self.assertTrue(all(torch.equal(buffers[n], b) for n, b in model.named_buffers()))
        for name, parameter in model.named_parameters():
            expected = grads[name]
            self.assertEqual(expected is None, parameter.grad is None)
            if expected is not None:
                self.assertTrue(torch.equal(expected, parameter.grad))
        self.assertEqual(modes, tuple(module.training for module in model.modules()))
        self.assertTrue(_nested_equal(optimizer_state, optimizer.state_dict()))
        self.assertTrue(_nested_equal(scheduler_state, scheduler.state_dict()))
        self.assertTrue(_nested_equal(precision_state, controller.state_dict()))
        self.assertEqual(precision_phase, controller.phase)
        self.assertTrue(_rng_equal(rng, capture_rng_state((loader.generator,))))
        self.assertEqual(perturbation_active, model._sample_fg_perturbation_active)

    def test_failure_restores_state_and_propagates_original_exception(self):
        model = _PurityModel()
        model.weight.grad = torch.full_like(model.weight, 8.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        controller = PrecisionController("fp32")
        service, loader = _service(
            model, _PurityDataset(fail_at=2), controller=controller
        )
        params = tuple(item.detach().clone() for item in model.parameters())
        buffers = tuple(item.detach().clone() for item in model.buffers())
        grad = model.weight.grad.detach().clone()
        modes = tuple(module.training for module in model.modules())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        precision_state = copy.deepcopy(controller.state_dict())
        precision_phase = controller.phase
        rng = capture_rng_state((loader.generator,))

        with self.assertRaisesRegex(RuntimeError, "intentional purity fixture failure"):
            service.compute(optimizer_step=2, purpose="diagnostic")

        self.assertTrue(all(torch.equal(a, b) for a, b in zip(params, model.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(buffers, model.buffers())))
        self.assertTrue(torch.equal(grad, model.weight.grad))
        self.assertIsNone(model.bias.grad)
        self.assertEqual(modes, tuple(module.training for module in model.modules()))
        self.assertTrue(_nested_equal(optimizer_state, optimizer.state_dict()))
        self.assertTrue(_nested_equal(scheduler_state, scheduler.state_dict()))
        self.assertTrue(_nested_equal(precision_state, controller.state_dict()))
        self.assertEqual(precision_phase, controller.phase)
        self.assertTrue(_rng_equal(rng, capture_rng_state((loader.generator,))))

    def test_eval_mode_is_preserved_for_synthetic_observational_query(self):
        model = _PurityModel()
        model.eval()
        modes = tuple(module.training for module in model.modules())
        service, _ = _service(model)
        result = service.compute(optimizer_step=0, purpose="diagnostic")
        self.assertTrue(result.gradient.is_finite())
        self.assertEqual(modes, tuple(module.training for module in model.modules()))

    def test_repeated_query_is_stateless_and_deterministic(self):
        model = _PurityModel()
        service, loader = _service(model)
        outside = capture_rng_state((loader.generator,))
        first = service.compute(optimizer_step=5, purpose="diagnostic")
        middle = capture_rng_state((loader.generator,))
        second = service.compute(optimizer_step=5, purpose="diagnostic")
        after = capture_rng_state((loader.generator,))
        for left, right in zip(first.gradient, second.gradient):
            self.assertTrue(torch.equal(left, right))
        first_meta = first.metadata.as_dict()
        second_meta = second.metadata.as_dict()
        first_meta.pop("elapsed_s")
        second_meta.pop("elapsed_s")
        self.assertEqual(first_meta, second_meta)
        self.assertTrue(_rng_equal(outside, middle))
        self.assertTrue(_rng_equal(outside, after))

    def test_normal_training_loader_continuation_is_observationally_identical(self):
        model = _PurityModel()
        normal_generator = torch.Generator(device="cpu")
        normal_generator.manual_seed(333)
        normal_dataset = _PurityDataset()
        normal_loader = DataLoader(
            normal_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            generator=normal_generator,
        )
        start = capture_rng_state((normal_generator,))

        control_iterator = iter(normal_loader)
        control_a = next(control_iterator)
        control_a_result = _batch_loss_and_gradient(model, control_a)
        control_b = next(control_iterator)
        control_b_result = _batch_loss_and_gradient(model, control_b)

        restore_rng_state(start)
        query_iterator = iter(normal_loader)
        query_a = next(query_iterator)
        query_a_result = _batch_loss_and_gradient(model, query_a)
        service, _ = _service(model)
        service.compute(optimizer_step=0, purpose="diagnostic")
        query_b = next(query_iterator)
        query_b_result = _batch_loss_and_gradient(model, query_b)

        self.assertEqual(list(control_a["sample_id"]), list(query_a["sample_id"]))
        self.assertEqual(list(control_b["sample_id"]), list(query_b["sample_id"]))
        self.assertTrue(torch.equal(control_a["img"], query_a["img"]))
        self.assertTrue(torch.equal(control_b["img"], query_b["img"]))
        self.assertEqual(control_a_result[0], query_a_result[0])
        self.assertEqual(control_b_result[0], query_b_result[0])
        for expected, actual in zip(control_b_result[1], query_b_result[1]):
            self.assertTrue(torch.equal(expected, actual))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_amp_scaler_and_cuda_state_are_unchanged(self):
        model = _PurityModel(device="cuda")
        scaler = GradScaler(init_scale=128.0, growth_interval=1000)
        controller = PrecisionController("amp", scaler=scaler)
        service, loader = _service(model, controller=controller)
        scaler_before = copy.deepcopy(controller.state_dict())
        phase_before = controller.phase
        rng_before = capture_rng_state((loader.generator,))
        result = service.compute(optimizer_step=1, purpose="diagnostic")
        self.assertTrue(result.gradient.is_finite())
        self.assertTrue(_nested_equal(scaler_before, controller.state_dict()))
        self.assertEqual(phase_before, controller.phase)
        self.assertTrue(_rng_equal(rng_before, capture_rng_state((loader.generator,))))


if __name__ == "__main__":
    unittest.main()
