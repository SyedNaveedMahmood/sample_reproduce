import tempfile
import unittest
from pathlib import Path

import torch

from dassl.config import get_cfg_default
from train import extend_cfg
from sample_fg.scheduler_compat import build_lr_scheduler_compat


class SchedulerCompatibilityTest(unittest.TestCase):
    def _cfg(self):
        cfg = get_cfg_default()
        extend_cfg(cfg)
        cfg.merge_from_file("configs/trainers/CoOp/vit_b16_ctxv1.yaml")
        cfg.defrost()
        cfg.OPTIM.MAX_EPOCH = 4
        cfg.freeze()
        return cfg

    def test_constant_warmup_then_cosine_epoch_trace_and_safe_state(self):
        cfg = self._cfg()
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=cfg.OPTIM.LR)
        scheduler = build_lr_scheduler_compat(optimizer, cfg.OPTIM)

        trace = [optimizer.param_groups[0]["lr"]]
        wrapper_epochs = [scheduler.last_epoch]
        successor_epochs = [scheduler.successor.last_epoch]
        for _ in range(3):
            optimizer.step()
            scheduler.step()
            trace.append(optimizer.param_groups[0]["lr"])
            wrapper_epochs.append(scheduler.last_epoch)
            successor_epochs.append(scheduler.successor.last_epoch)

        expected = [
            cfg.OPTIM.WARMUP_CONS_LR,
            cfg.OPTIM.LR,
            cfg.OPTIM.LR * (1.0 + 2.0 ** -0.5) / 2.0,
            cfg.OPTIM.LR / 2.0,
        ]
        for observed, reference in zip(trace, expected):
            self.assertAlmostEqual(observed, reference, places=12)
        self.assertEqual(wrapper_epochs, [0, 1, 1, 1])
        self.assertEqual(successor_epochs, [0, 0, 1, 2])

        state = scheduler.state_dict()
        self.assertNotIn("successor", state)
        self.assertIn("_successor_state_dict", state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scheduler.pt"
            torch.save(state, path)
            loaded = torch.load(path)
        self.assertEqual(loaded["last_epoch"], state["last_epoch"])

        parameter2 = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer2 = torch.optim.SGD([parameter2], lr=cfg.OPTIM.LR)
        scheduler2 = build_lr_scheduler_compat(optimizer2, cfg.OPTIM)
        scheduler2.load_state_dict(loaded)
        self.assertEqual(scheduler2.last_epoch, scheduler.last_epoch)
        self.assertEqual(
            scheduler2.successor.last_epoch, scheduler.successor.last_epoch
        )


if __name__ == "__main__":
    unittest.main()
