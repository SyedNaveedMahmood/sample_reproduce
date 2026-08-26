# SAMPLe Reproduction and Follow-up Analysis

This repository contains an independent reproduction and follow-up analysis of **SAMPLe: A Sharpness Aware Minimization based Optimizer for Prompt Learning in Vision-Language Models** (Rajoli et al., 2026), built on the official CoOp/CoCoOp codebase.

The project focuses on reproducing the reported base-to-new generalization results and studying the role of SAMPLe's EMA-based global-gradient approximation. In addition to CoOp, SAM, and paper-style SAMPLe, the repository includes exact and periodic full-gradient variants plus diagnostics for estimator fidelity, optimization geometry, prompt-space sharpness, and open-world evaluation.

Paper: https://arxiv.org/abs/2607.05727

## Reproduction setup

- **Backbone:** CLIP ViT-B/16
- **Datasets:** DTD and EuroSAT
- **Protocol:** 16-shot base-to-new generalization
- **Seeds:** 1, 2, 3
- **Prompt initialization:** `a photo of a`
- **Learnable context tokens:** 4
- **Training:** 200 epochs, SGD, cosine schedule
- **Precision:** FP16

The main reproduction configuration is in `configs/sample_fg/paper_reproduction.yaml`.

## Reproduction results

Values below are accuracy percentages. Reproduction results are mean ± sample standard deviation across seeds 1–3.

### DTD

| Method | Paper New | Repro New | Paper HM | Repro HM |
|---|---:|---:|---:|---:|
| CoOp | 41.18 | 42.43 ± 1.99 | 54.24 | 55.38 ± 1.76 |
| SAM | 56.52 | 46.46 ± 4.05 | 65.18 | 58.77 ± 3.33 |
| SAMPLe | 63.04 | 45.49 ± 0.94 | 69.77 | 58.32 ± 0.83 |

### EuroSAT

| Method | Paper New | Repro New | Paper HM | Repro HM |
|---|---:|---:|---:|---:|
| CoOp | 54.74 | 56.28 ± 2.51 | 68.69 | 69.94 ± 2.14 |
| SAM | 61.23 | 65.74 ± 4.73 | 71.87 | 75.99 ± 3.68 |
| SAMPLe | 75.43 | 54.91 ± 7.44 | 82.44 | 68.19 ± 5.92 |

CoOp reproduces closely on both datasets. The reported SAMPLe gain on novel classes was not recovered under this setup; the discrepancy is concentrated mainly on New-class accuracy rather than Base accuracy.

## Follow-up experiments

The repository also contains controlled experiments that test whether the EMA approximation itself explains the generalization gap.

- On DTD, the EMA global-gradient proxy had mean cosine similarity **0.145 ± 0.007** with the exact gradient and relative L2 error **3.12 ± 0.09**.
- On EuroSAT, the corresponding mean cosine similarity was **0.237 ± 0.043**.
- In a DTD seed-1 estimator substitution test, New/HM improved from **46.26/58.71** with paper-style EMA to **48.07/60.08** with an exact full gradient and **52.17/62.86** with periodic exact refresh (`K=8`). None recovered the reported SAMPLe result.
- On EuroSAT, SAMPLe produced the lowest sampled prompt-space sharpness across the three seeds while still giving lower New accuracy than SAM and CoOp, suggesting that prompt-space flatness alone is not sufficient to explain novel-class generalization.

These follow-up results are exploratory where only a single seed was used and should not be interpreted as a replacement for the full multi-seed reproduction.

## Repository structure

- `sample_fg/` — CoOp/SAM/SAMPLe reproduction and gradient-estimator implementations
- `analysis/` — aggregation and diagnostic utilities
- `configs/sample_fg/` — reproduction and extension configurations
- `scripts/` — experiment, validation, and analysis scripts
- `tests/` — implementation and protocol tests
- `train_sample_fg.py` — main reproduction entry point
- `train_sample_fg_extension.py` — extension entry point

For the available command-line options:

```bash
python train_sample_fg.py --help
python train_sample_fg_extension.py --help
```

## Installation

This code builds on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) and the original [CoOp](https://github.com/KaiyangZhou/CoOp) implementation.

Install Dassl first, then install the additional requirements:

```bash
pip install -r requirements.txt
```

The original CoOp/CoCoOp dataset and execution notes are retained in `DATASETS.md`, `COOP.md`, and `COCOOP.md`.

## Acknowledgements

This repository is derived from the official CoOp/CoCoOp codebase by Kaiyang Zhou and collaborators. Please cite the original CoOp, CoCoOp, and SAMPLe papers when using the corresponding components of this repository.

### CoOp

```bibtex
@article{zhou2022coop,
  title={Learning to Prompt for Vision-Language Models},
  author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  journal={International Journal of Computer Vision},
  year={2022}
}
```

### CoCoOp

```bibtex
@inproceedings{zhou2022cocoop,
  title={Conditional Prompt Learning for Vision-Language Models},
  author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
