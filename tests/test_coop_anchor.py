import unittest

import torch

from sample_fg.coop_anchor import (
    audit_prompt_only_training,
    run_bounded_coop_steps,
)


class _PromptLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = torch.nn.Parameter(torch.ones(2, 3))


class _TinyAnchorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = _PromptLearner()
        self.image_encoder = torch.nn.Module()
        self.image_encoder.proj = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.text_encoder = torch.nn.Module()
        self.text_encoder.transformer = torch.nn.Linear(1, 1)
        self.text_encoder.positional_embedding = torch.nn.Parameter(
            torch.ones(1), requires_grad=False
        )
        self.text_encoder.text_projection = torch.nn.Parameter(
            torch.ones(1), requires_grad=False
        )
        self.text_encoder.ln_final = torch.nn.LayerNorm(1)
        self.logit_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        for name, parameter in self.named_parameters():
            if name != "prompt_learner.ctx":
                parameter.requires_grad_(False)


class _TinyTrainer:
    def __init__(self):
        self.model = _TinyAnchorModel()
        self.optim = torch.optim.SGD(self.model.prompt_learner.parameters(), lr=0.1)
        self.train_loader_x = [{"value": float(i + 1)} for i in range(5)]
        self.batch_idx = 0
        self.num_batches = 0

    def set_model_mode(self, _mode):
        return None

    def get_current_lr(self):
        return self.optim.param_groups[0]["lr"]

    def forward_backward(self, batch):
        self.optim.zero_grad()
        loss = self.model.prompt_learner.ctx.sum() * batch["value"]
        loss.backward()
        self.optim.step()
        return {"loss": float(loss.item()), "acc": 0.0}


class CoOpAnchorControlTest(unittest.TestCase):
    def test_prompt_only_audit_and_optimizer_membership(self):
        trainer = _TinyTrainer()
        audit = audit_prompt_only_training(trainer.model, trainer.optim)
        self.assertEqual(
            [item["name"] for item in audit["trainable_parameters"]],
            ["prompt_learner.ctx"],
        )
        self.assertEqual(audit["trainable_numel"], 6)

    def test_max_optimizer_steps_is_exact(self):
        trainer = _TinyTrainer()
        records = run_bounded_coop_steps(trainer, max_optimizer_steps=3)
        self.assertEqual(len(records), 3)
        self.assertEqual([item["optimizer_step"] for item in records], [1, 2, 3])
        self.assertEqual([item["batch_index_zero_based"] for item in records], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
