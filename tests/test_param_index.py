import unittest

import torch

from sample_fg.param_index import (
    FINGERPRINT_SCHEMA,
    ParamIndex,
    ParamIndexError,
    ParamIndexMismatchError,
)


class _MixedModel(torch.nn.Module):
    def __init__(self, first_shape=(2, 3)):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(first_shape))
        self.frozen = torch.nn.Parameter(torch.zeros(5), requires_grad=False)
        self.block = torch.nn.Linear(4, 2, bias=False)


class _RenamedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.renamed = torch.nn.Parameter(torch.zeros(2, 3))
        self.frozen = torch.nn.Parameter(torch.zeros(5), requires_grad=False)
        self.block = torch.nn.Linear(4, 2, bias=False)


class _OrderedModel(torch.nn.Module):
    def __init__(self, reverse=False):
        super().__init__()
        if reverse:
            self.register_parameter("right", torch.nn.Parameter(torch.zeros(2)))
            self.register_parameter("left", torch.nn.Parameter(torch.zeros(2)))
        else:
            self.register_parameter("left", torch.nn.Parameter(torch.zeros(2)))
            self.register_parameter("right", torch.nn.Parameter(torch.zeros(2)))


class ParamIndexTest(unittest.TestCase):
    def test_canonical_metadata_and_frozen_exclusion(self):
        model = _MixedModel()
        index = ParamIndex.from_model(model)

        self.assertEqual(index.names, ("first", "block.weight"))
        self.assertEqual([entry.shape for entry in index], [(2, 3), (2, 4)])
        self.assertEqual([entry.numel for entry in index], [6, 8])
        self.assertEqual(index.total_numel, 14)
        self.assertNotIn("frozen", index.names)
        self.assertEqual(index.fingerprint_schema, FINGERPRINT_SCHEMA)
        metadata = index.to_metadata()
        self.assertEqual(metadata["total_numel"], 14)
        self.assertEqual(metadata["parameters"][0]["dtype"], "torch.float32")
        self.assertEqual(metadata["parameters"][0]["device"], "cpu")

    def test_fingerprint_is_deterministic(self):
        first = ParamIndex.from_model(_MixedModel())
        second = ParamIndex.from_model(_MixedModel())
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.fingerprint_payload, second.fingerprint_payload)
        first.assert_compatible(second)

    def test_fingerprint_changes_with_name_shape_or_order(self):
        reference = ParamIndex.from_model(_MixedModel())
        renamed = ParamIndex.from_model(_RenamedModel())
        reshaped = ParamIndex.from_model(_MixedModel(first_shape=(3, 2)))
        ordered = ParamIndex.from_model(_OrderedModel(reverse=False))
        reversed_order = ParamIndex.from_model(_OrderedModel(reverse=True))

        self.assertNotEqual(reference.fingerprint, renamed.fingerprint)
        self.assertNotEqual(reference.fingerprint, reshaped.fingerprint)
        self.assertEqual(set(ordered.names), set(reversed_order.names))
        self.assertNotEqual(ordered.fingerprint, reversed_order.fingerprint)

    def test_duplicate_parameter_identity_is_rejected(self):
        model = torch.nn.Module()
        shared = torch.nn.Parameter(torch.ones(1))
        model.register_parameter("first", shared)
        model.register_parameter("alias", shared)
        with self.assertRaises(ParamIndexError):
            ParamIndex.from_model(model)

    def test_explicit_model_revalidation_detects_trainable_change(self):
        model = _MixedModel()
        index = ParamIndex.from_model(model)
        model.frozen.requires_grad_(True)
        with self.assertRaises(ParamIndexMismatchError):
            index.assert_matches_model(model)


if __name__ == "__main__":
    unittest.main()
