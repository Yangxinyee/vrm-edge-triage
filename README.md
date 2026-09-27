# Confidence-Gated Cloud-Edge Cascade Triage via Variational Risk Minimization for Medical Imaging

Official code for our paper in **Smart Health** (2026), presented as an **oral** at **IEEE/ACM CHASE 2026**.

[![Paper](https://img.shields.io/badge/Paper-Smart%20Health%202026-blue)](https://doi.org/10.1016/j.smhl.2026.100689)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.smhl.2026.100689-green)](https://doi.org/10.1016/j.smhl.2026.100689)

**Xinye Yang**, Zhusi Zhong, Scott Collins, Michael Bernstein, Grayson Baird, Terrence Healey, Michael Atalay, Mahesh Jayaraman, Xuyu Wang, Zhicheng Jiao

VRM trains a lightweight, image-only chest X-ray triage model for edge deployment by distilling a multimodal (image + report) teacher, with teacher targets marginalized over Monte Carlo report variants sampled offline from a large vision-language model. In the paper, the edge model sits in a confidence-gated cloud-edge cascade for medical imaging triage.

**Related code:** [Q-DISTILL](https://github.com/Yangxinyee/Q-DISTILL), self-supervised Q-Former distillation for image-only chest X-ray triage, from the same line of work. A mirror of this code is also available at [CHASE-2026-vrm/vrm-edge-triage](https://github.com/CHASE-2026-vrm/vrm-edge-triage).

This repository contains the core implementation of **Variational Risk Minimization (VRM)** for chest X-ray triage:
- multimodal teacher (BiomedCLIP + LQA),
- image-only student distilled from marginalized teacher targets,
- offline LVLM sampling for Monte Carlo report variants.

## What Is Included

```
vrm-edge-triage/
├── models/
│   ├── teacher.py      # Teacher model (kept unchanged)
│   ├── student.py      # Student model with EVA-X encoder transfer
│   └── losses.py       # VRM losses
├── scripts/
│   ├── generate_variational_samples.py
│   ├── train_teacher.py
│   ├── train_student_vrm.py
│   └── evaluate.py
└── requirements.txt
```

## Student Encoder Transfer (EVA-X)

By default, the student uses an EVA-family tiny backbone and tries to initialize
the encoder from:

`checkpoints/eva_x_tiny_patch16_merged520k_mim.pt`

Teacher behavior is unchanged. During distillation, the teacher is frozen.

## Data Layout

```
data/mimic_cxr/
├── images/
├── reports/
├── labels.csv
├── train_list.txt
├── val_list.txt
└── test_list.txt
```

`labels.csv` must contain:

```csv
id,urgency_label
sample_0001,0
sample_0002,1
```

## Minimal Run Commands

1) Generate variational samples (K=5):

```bash
python scripts/generate_variational_samples.py \
  --data_root data/mimic_cxr \
  --split train \
  --k_samples 5 \
  --output_file data/mimic_cxr/train_variational_samples.json
```

2) Train teacher:

```bash
python scripts/train_teacher.py \
  --data_root data/mimic_cxr \
  --output_dir outputs/teacher
```

3) Distill student (teacher frozen):

```bash
python scripts/train_student_vrm.py \
  --data_root data/mimic_cxr \
  --teacher_checkpoint outputs/teacher/best_model.pt \
  --samples_file data/mimic_cxr/train_variational_samples.json \
  --student_backbone_checkpoint checkpoints/eva_x_tiny_patch16_merged520k_mim.pt \
  --output_dir outputs/student_vrm
```

4) Evaluate student:

```bash
python scripts/evaluate.py \
  --model student \
  --checkpoint outputs/student_vrm/best_model.pt \
  --data_root data/mimic_cxr \
  --student_backbone_checkpoint checkpoints/eva_x_tiny_patch16_merged520k_mim.pt
```

## Citation

If you use this code, please cite:

```bibtex
@article{yang2026vrm,
  title   = {Confidence-gated cloud-edge cascade triage via variational risk minimization for medical imaging},
  author  = {Yang, Xinye and Zhong, Zhusi and Collins, Scott and Bernstein, Michael and Baird, Grayson and Healey, Terrence and Atalay, Michael and Jayaraman, Mahesh and Wang, Xuyu and Jiao, Zhicheng},
  journal = {Smart Health},
  volume  = {41},
  pages   = {100689},
  year    = {2026},
  doi     = {10.1016/j.smhl.2026.100689}
}
```
