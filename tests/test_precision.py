import unittest
from unittest import mock

import torch
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.precision import (
    PrecisionController,
    PrecisionError,
    PrecisionNumericalError,
    PrecisionStateError,
)


class _TinyModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(
                [[0.25, -0.5, 0.75], [-0.125, 0.375, 0.625]],
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, value):
        return value @ self.weight.t()


def _capture_toy(mode, device, model_dtype, scale=None):
    model = _TinyModel(device=device, dtype=model_dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = None
    if mode == "amp":
        scaler = GradScaler(init_scale=scale, growth_interval=1000)
    controller = PrecisionController(mode, scaler=scaler)
    index = ParamIndex.from_model(model)
    values = torch.tensor(
        [[1.0, -2.0, 0.5], [-0.25, 0.75, 2.0]],
        device=device,
        dtype=model_dtype,
    )
    targets = torch.tensor(
        [[0.5, -1.0], [1.25, 0.25]],
        device=device,
        dtype=model_dtype,
    )
    controller.begin(optimizer)
    with controller.autocast_context():
        output = model(values)
        loss = F.mse_loss(output, targets)
    controller.backward(loss)
    live_before = tuple(parameter.grad.detach().clone() for parameter in index.parameters)
    capture = controller.capture_gradients(index, optimizer)
    return model, optimizer, controller, index, capture, live_before, loss


class PrecisionControllerTest(unittest.TestCase):
    def test_fp32_capture_matches_live_gradient_and_never_steps(self):
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        controller = PrecisionController("fp32")
        index = ParamIndex.from_model(model)
        values = torch.tensor([[1.0, -2.0, 0.5]])
        target = torch.tensor([[0.5, -1.0]])
        original_step = optimizer.step
        with mock.patch.object(optimizer, "step", wraps=original_step) as step_mock:
            controller.begin(optimizer)
            with controller.autocast_context():
                loss = F.mse_loss(model(values), target)
            controller.backward(loss)
            live = model.weight.grad.detach().clone()
            capture = controller.capture_gradients(index, optimizer)
            step_mock.assert_not_called()

        self.assertIsNone(controller.scaler)
        self.assertFalse(capture.scaling_active)
        self.assertFalse(capture.authoritative_unscale_performed)
        self.assertEqual(capture.state[0].dtype, torch.float32)
        self.assertEqual(capture.state[0].device.type, "cpu")
        self.assertTrue(torch.equal(capture.state[0], live.float()))
        self.assertTrue(capture.state.is_finite())
        self.assertTrue(
            all(not item.requires_grad and item.grad_fn is None for item in capture.state)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_native_fp16_capture_is_owned_fp32_on_cuda(self):
        model, optimizer, _, index, capture, live_before, _ = _capture_toy(
            "fp16", "cuda", torch.float16
        )
        self.assertEqual(model.weight.dtype, torch.float16)
        self.assertEqual(live_before[0].dtype, torch.float16)
        self.assertEqual(capture.live_dtypes_after_unscale, ("torch.float16",))
        self.assertEqual(capture.state[0].dtype, torch.float32)
        self.assertEqual(capture.state[0].device.type, "cuda")
        snapshot = capture.state.clone()
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(index[0].parameter.grad)
        self.assertTrue(torch.equal(capture.state[0], snapshot[0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_amp_unscaled_capture_is_independent_of_scaler_magnitude(self):
        low = _capture_toy("amp", "cuda", torch.float32, scale=8.0)
        # Keep both scales inside this toy MSE graph's finite FP16 range. The
        # 128x separation remains large enough to expose a missing unscale.
        high = _capture_toy("amp", "cuda", torch.float32, scale=1024.0)
        low_capture = low[4]
        high_capture = high[4]
        self.assertEqual(low_capture.scale, 8.0)
        self.assertEqual(high_capture.scale, 1024.0)
        self.assertTrue(low_capture.authoritative_unscale_performed)
        self.assertTrue(high_capture.authoritative_unscale_performed)
        torch.testing.assert_close(
            low_capture.state[0],
            high_capture.state[0],
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(low_capture.state[0].dtype, torch.float32)
        self.assertEqual(high_capture.state[0].dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_amp_unscale_and_scaler_transition_are_exactly_once(self):
        model = _TinyModel(device="cuda")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = GradScaler(init_scale=128.0, growth_interval=1000)
        controller = PrecisionController("amp", scaler=scaler)
        index = ParamIndex.from_model(model)
        values = torch.tensor([[1.0, -2.0, 0.5]], device="cuda")
        target = torch.tensor([[0.5, -1.0]], device="cuda")
        controller.begin(optimizer)
        with controller.autocast_context():
            loss = F.mse_loss(model(values), target)
        controller.backward(loss)

        with mock.patch.object(scaler, "unscale_", wraps=scaler.unscale_) as unscale:
            capture = controller.capture_gradients(index, optimizer)
            unscale.assert_called_once_with(optimizer)
            with self.assertRaises(PrecisionStateError):
                controller.capture_gradients(index, optimizer)
            unscale.assert_called_once()

        controller.install_logical_gradients(index, capture.state)
        with mock.patch.object(scaler, "step", wraps=scaler.step) as scaler_step:
            with mock.patch.object(scaler, "update", wraps=scaler.update) as update:
                result = controller.step(optimizer)
                scaler_step.assert_called_once_with(optimizer)
                update.assert_called_once_with()
                with self.assertRaises(PrecisionStateError):
                    controller.step(optimizer)
                scaler_step.assert_called_once()
                update.assert_called_once()
        self.assertFalse(result.scaler_step_skipped)

    def test_installation_preserves_state_and_casts_to_live_dtype(self):
        model, _, controller, index, capture, _, _ = _capture_toy(
            "fp32", "cpu", torch.float32
        )
        logical = capture.state.scale(0.5)
        snapshot = logical.clone()
        installed_dtypes = controller.install_logical_gradients(index, logical)
        self.assertEqual(installed_dtypes, ("torch.float32",))
        self.assertTrue(torch.equal(model.weight.grad, logical[0]))
        self.assertTrue(torch.equal(logical[0], snapshot[0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_fp16_installation_casts_owned_logical_state_to_live_dtype(self):
        model, _, controller, index, capture, _, _ = _capture_toy(
            "fp16", "cuda", torch.float16
        )
        logical = capture.state.scale(0.5)
        snapshot = logical.clone()
        installed_dtypes = controller.install_logical_gradients(index, logical)
        self.assertEqual(installed_dtypes, ("torch.float16",))
        self.assertEqual(model.weight.grad.dtype, torch.float16)
        torch.testing.assert_close(
            model.weight.grad.float(), logical[0], rtol=1e-3, atol=1e-4
        )
        self.assertTrue(torch.equal(logical[0], snapshot[0]))

    def test_clear_live_grad_does_not_mutate_capture(self):
        _, optimizer, _, index, capture, _, _ = _capture_toy(
            "fp32", "cpu", torch.float32
        )
        snapshot = capture.state.clone()
        optimizer.zero_grad(set_to_none=True)
        self.assertIsNone(index[0].parameter.grad)
        self.assertTrue(torch.equal(capture.state[0], snapshot[0]))

    def test_nonfinite_loss_and_live_gradient_fail_without_repair(self):
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        controller = PrecisionController("fp32")
        index = ParamIndex.from_model(model)
        controller.begin(optimizer)
        bad_loss = model(torch.ones(1, 3)).sum() * float("nan")
        with self.assertRaises(PrecisionNumericalError):
            controller.backward(bad_loss)

        controller = PrecisionController("fp32")
        controller.begin(optimizer)
        loss = model(torch.ones(1, 3)).sum()
        controller.backward(loss)
        model.weight.grad.fill_(float("inf"))
        with self.assertRaises(PrecisionNumericalError):
            controller.capture_gradients(index, optimizer)
        self.assertTrue(torch.isinf(model.weight.grad).all())

    def test_configuration_sequence_and_optimizer_membership_fail_fast(self):
        with self.assertRaises(PrecisionError):
            PrecisionController("unknown")
        with self.assertRaises(PrecisionError):
            PrecisionController("fp32", scaler=mock.Mock())

        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        controller = PrecisionController("fp32")
        with self.assertRaises(PrecisionStateError):
            controller.backward(torch.tensor(1.0, requires_grad=True))
        controller.begin(optimizer)
        loss = model(torch.ones(1, 3)).sum()
        controller.backward(loss)
        extra = torch.nn.Parameter(torch.ones(1))
        wrong_optimizer = torch.optim.SGD([extra], lr=0.1)
        with self.assertRaises(PrecisionStateError):
            controller.capture_gradients(ParamIndex.from_model(model), wrong_optimizer)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_amp_scaler_state_roundtrip(self):
        _, _, controller, _, _, _, _ = _capture_toy(
            "amp", "cuda", torch.float32, scale=512.0
        )
        payload = controller.state_dict()
        restored = PrecisionController(
            "amp", scaler=GradScaler(init_scale=2.0, growth_interval=1000)
        )
        restored.load_state_dict(payload)
        self.assertEqual(restored.scaler.get_scale(), 512.0)
        with self.assertRaises(PrecisionError):
            PrecisionController("fp32").load_state_dict(payload)


if __name__ == "__main__":
    unittest.main()
