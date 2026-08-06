[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python)](https://www.python.org/)

# HEAL-Net

**A Hierarchical Knowledge Distillation and Neural Architecture Search Framework for Lightweight Multimodal Medical Diagnosis on Edge Devices**

## Overview

Traditional multimodal medical diagnosis models rely on large network architectures, leading to high computational complexity that prevents deployment on resource-constrained edge devices. Moreover, these models often fail when encountering modality missing in real-world clinical environments.

**HEAL-Net** addresses these challenges with three core innovations:

| Module | Full Name | Role |
|--------|-----------|------|
| **HD-NAS** | Hardware-aware Differentiable Neural Architecture Search | Evolves optimal lightweight fusion topology with hardware penalty |
| **HKD** | Hierarchical Knowledge Distillation | Transfers cross-modal covariance knowledge from teacher to student |
| **DFC** | Dynamic Feature Compensation | Reconstructs missing modality features via lightweight generator |

**Key Results (MIMIC Dataset):**

| Metric | HEAL-Net | Best Baseline | Improvement |
|--------|----------|---------------|-------------|
| Accuracy | **89.85% +/- 0.28** | 87.3% +/- 0.38 | +2.55% |
| Parameters | **8.8M** | 86.4M | **~10x smaller** |
| FLOPs | **2.2G** | 28.1G | **~13x fewer** |
| Inference | **4.8ms** | 42.3ms | **~9x faster** |
| 90% Missing Acc | **80.15%** | 74.82% | +5.33% |

> Best baseline is **AdaCoMed**, the strongest prior model on this benchmark (see [Results](#results)).

## Environment Setup

**Prerequisites:**
- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.3 (for GPU) or MPS (Apple Silicon)

**Installation:**

```bash
git clone https://github.com/your-username/HEAL-Net.git
cd HEAL-Net

conda create -n healnet python=3.9 -y
conda activate healnet

conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

**Key Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0.0 | Deep learning framework |
| torchvision | >= 0.15.0 | Image transforms and models |
| fvcore | >= 0.1.5 | FLOPs profiling |
| thop | >= 0.1.1 | MACs computation |
| scikit-learn | >= 1.2.0 | Evaluation metrics |
| tqdm | >= 4.66.0 | Progress bars |

## Dataset Preparation

### MIMIC Multimodal Dataset

The dataset combines MIMIC-IV clinical records with MIMIC-CXR chest X-ray images.

**Expected structure:**

```
data/mimic/
+-- train/
|   +-- images/           # Chest X-ray PNG images
|   +-- ehr.npy           # EHR features (N, 12)
|   +-- labels.npy        # Labels (N,) values 0-4
|   +-- split.json
+-- val/
+-- test/
```

**Disease Classes (5):**

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | Normal | No pathology detected |
| 1 | Cardiac Hypertrophy | Enlarged heart |
| 2 | Pulmonary Edema | Fluid in lungs |
| 3 | Pulmonary Consolidation | Lung tissue solidification |
| 4 | Pleural Effusion | Fluid around lungs |

**EHR Features (12):**
Core laboratory values including glucose, BUN, creatinine, sodium, potassium, chloride, bicarbonate, hematocrit, WBC, platelet, magnesium, calcium.

## Training

### Full Pipeline (End-to-End)

```bash
python run.py --mode train --stage all --config configs/config.yaml
```

### Step-by-Step Training

#### Stage 1: Architecture Search (HD-NAS)

```bash
python run.py --mode train --stage search --config configs/config.yaml
```

#### Stage 2: Knowledge Distillation (HKD)

```bash
python run.py --mode train --stage distill --teacher_path ./weights/teacher_pretrained.pth
```

#### Stage 3: Dynamic Feature Compensation (DFC)

```bash
python run.py --mode train --stage compensate
```

**Arguments:**

| Argument | Default | Choices | Description |
|----------|---------|---------|-------------|
| `--mode` | `train` | `train`, `test`, `ablation` | Running mode |
| `--stage` | `all` | `search`, `distill`, `compensate`, `all` | Training stage |
| `--config` | `configs/config.yaml` | -- | Configuration file |
| `--teacher_path` | `None` | -- | Pretrained teacher model path |
| `--model_path` | `None` | -- | Student model path (for test) |
| `--num_seeds` | `5` (from config) | -- | Seeds for ablation experiments |

**Training Details:**
- **Optimizer:** AdamW (lr=0.001, weight_decay=0.0001)
- **Scheduler:** Cosine Annealing (T_max=200, min_lr=1e-6)
- **Early Stopping:** Patience=20 epochs
- **AMP:** Enabled for mixed precision training
- **Gradient Clipping:** max_norm=1.0
- **Batch Size:** 64

## Testing and Evaluation

### Standard Evaluation

```bash
python test.py --model_path ./weights/heal_net_compensate.pth --config configs/config.yaml
```

**Outputs:**
- `results/metrics.json` -- Accuracy, F1, AUC, Params, FLOPs
- Console output with all metrics

### Efficiency Profiling

```bash
python efficiency_eval.py --config configs/config.yaml --model_path ./weights/heal_net_compensate.pth
```

**Outputs:**
- Parameter count (total and trainable)
- FLOPs (GFLOPs)
- Inference latency (mean +/- std ms)
- Results saved to `results/efficiency.json`

## Ablation Guide

Run ablation experiments to isolate each module's contribution:

```bash
python run.py --mode ablation --num_seeds 5 --config configs/config.yaml
```

This evaluates all combinations:

| Variant | HD-NAS | HKD | DFC | Params (M) | FLOPs (G) | Acc (ideal) | Acc (50% missing) |
|---------|--------|-----|-----|------------|-----------|-------------|-------------------|
| Base | x | x | x | 18.5 | 5.6 | 84.23 +/- 0.65 | 61.34 +/- 1.12 |
| V1 | v | x | x | 8.3 | 2.1 | 85.11 +/- 0.58 | 63.52 +/- 0.95 |
| V2 | x | v | x | 18.5 | 5.6 | 86.45 +/- 0.52 | 62.88 +/- 1.05 |
| V3 | x | x | v | 19 | 5.7 | 84.35 +/- 0.61 | 80.12 +/- 0.76 |
| V4 | v | v | x | 8.3 | 2.1 | 89.76 +/- 0.32 | 67.84 +/- 0.88 |
| V5 | v | x | v | 8.8 | 2.2 | 85.20 +/- 0.55 | 80.45 +/- 0.73 |
| V6 | x | v | v | 19 | 5.7 | 86.58 +/- 0.49 | 82.35 +/- 0.68 |
| **Full** | **v** | **v** | **v** | **8.8** | **2.2** | **89.85 +/- 0.28** | **84.58 +/- 0.45** |

All metrics are mean +/- std over 5 independent random seeds.

**Outputs:**
- `results/experiment_summary.json` -- Cohen's d effect sizes, p-values

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | AdamW initial LR |
| Weight Decay | 0.0001 | L2 regularization |
| Batch Size | 64 | Training batch size |
| Epochs | 200 | Total training epochs |
| Temperature | 3.0 | Distillation temperature |
| gamma | 1.0 | Relation loss weight |
| beta | 0.5 | KL divergence weight |
| lambda | 0.05 | FLOPs penalty weight |
| rho | 0.1 | Reconstruction loss weight |
| SNR threshold | 0.3 | DFC mask threshold |
| NAS nodes | 4 | DAG search space nodes |

## Results

### Multi-Modal Diagnosis Performance

| Method | Acc (%) | F1 | AUC | Params (M) | FLOPs (G) | Latency (ms) |
|--------|---------|-----|-----|------------|-----------|--------------|
| MDF-Net | 81.8 +/- 0.61 | 79.6 +/- 0.72 | 0.901 | 14.5 | 4.2 | 8.7 |
| HyperFusion | 85.6 +/- 0.55 | 83.9 +/- 0.64 | 0.920 | 26.7 | 8.3 | 16.5 |
| EHR-KnowGen | 86.4 +/- 0.49 | 85.0 +/- 0.58 | 0.926 | 145.6 | 42.5 | 78.4 |
| MOFS | 87.2 +/- 0.45 | 85.7 +/- 0.55 | 0.930 | 210.3 | 65.4 | 115.6 |
| AdaCoMed | 87.3 +/- 0.38 | 86.1 +/- 0.47 | 0.931 | 86.4 | 28.1 | 42.3 |
| **HEAL-Net** | **89.85 +/- 0.28** | **89.1 +/- 0.33** | **0.948** | **8.8** | **2.2** | **4.8** |

### Robustness Under Modality Missing

| Missing Rate | MDF-Net | HyperFusion | EHR-KnowGen | MOFS | AdaCoMed | **HEAL-Net** |
|--------------|---------|-------------|-------------|------|----------|--------------|
| 0% | 81.8 | 85.6 | 86.4 | 87.2 | 87.3 | **89.85** |
| 10% | 81.24 | 85.16 | 86.02 | 86.85 | 86.94 | **89.12** |
| 30% | 74.56 | 78.42 | 80.57 | 81.29 | 84.25 | **87.35** |
| 50% | 65.31 | 69.55 | 72.14 | 73.66 | 80.51 | **84.58** |
| 70% | 58.12 | 62.18 | 65.39 | 66.82 | 77.34 | **82.41** |
| 90% | 51.45 | 54.36 | 58.71 | 60.14 | 74.82 | **80.15** |

*Accuracy (%) under increasing EHR-modality missing rates (paired with Table 2 in the paper).*

## Project Structure

```
HEAL-Net/
+-- README.md                          # This file
+-- requirements.txt                   # Python dependencies
+-- run.py                             # Unified entry (train/test/ablation)
+-- test.py                            # Standalone evaluation
+-- efficiency_eval.py                 # Hardware efficiency profiling
+-- models/
|   +-- __init__.py
|   +-- hd_nas.py                      # HD-NAS: hardware-aware architecture search
|   +-- hkd_module.py                  # HKD: cross-modal knowledge distillation
|   +-- dfc.py                         # DFC: dynamic feature compensation
|   +-- heal_net.py                    # HEAL-Net complete model
|   +-- teacher_model.py               # Teacher model (ViT-B/32 + TabNet)
|   +-- builder.py                     # Model factory function
+-- data/
|   +-- __init__.py
|   +-- dataset.py                     # MIMIC multimodal dataset
|   +-- synthetic.py                   # Synthetic data fallback (no-dataset quickstart)
|   +-- transforms.py                  # Data augmentation
+-- losses/
|   +-- __init__.py                    # Joint loss function
+-- engines/
|   +-- __init__.py
|   +-- trainer.py                     # Stage-aware training engine (search/distill/compensate)
|   +-- evaluator.py                   # Evaluation engine
+-- utils/
|   +-- __init__.py
|   +-- config.py                      # YAML config loader (scientific-notation aware)
|   +-- metrics.py                     # Evaluation metrics
|   +-- early_stopping.py              # Early stopping
|   +-- reproducibility.py             # Seed固定
|   +-- experiment_recorder.py         # Experiment recording
+-- configs/
|   +-- config.yaml                    # Hyperparameter configuration
+-- weights/                           # Model checkpoints (gitignore)
+-- results/                           # Evaluation results
+-- manifest.json                      # Model architecture manifest
```

### File Descriptions

| File | Responsibility |
|------|---------------|
| `run.py` | Pipeline orchestrator with --stage and --mode arguments |
| `test.py` | Model evaluation with metrics and efficiency |
| `efficiency_eval.py` | Hardware efficiency profiling (params, FLOPs, latency) |
| `models/hd_nas.py` | Differentiable search space with mixed operations |
| `models/hkd_module.py` | Cross-modal covariance alignment + KL distillation |
| `models/dfc.py` | SNR masking + cross-modal generator + gated fusion |
| `models/heal_net.py` | Complete HEAL-Net integrating all three modules |
| `models/teacher_model.py` | Teacher model: ViT-B/32 + TabNet + fusion |
| `models/builder.py` | Model factory with ablation support |
| `losses/__init__.py` | Joint loss: L_Distill + lambda*log(FLOPs) + rho*L_Recon |
| `engines/trainer.py` | Training loop with AMP, gradient clipping, early stopping |
| `engines/evaluator.py` | Test evaluation with metrics and efficiency |
| `utils/metrics.py` | Accuracy, F1, AUC, FLOPs computation |
| `utils/experiment_recorder.py` | Ablation recording with Cohen's d effect size |

## Citation

If you use this code, please cite:

```bibtex
@article{healnet2025,
  title={HEAL-Net: A Hierarchical Knowledge Distillation and Neural Architecture Search Framework for Lightweight Multimodal Medical Diagnosis on Edge Devices},
  author={},
  year={2025}
}
```

## License

This project is released for academic research use.

## Contact

For questions, please open an issue or contact the authors.
