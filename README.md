# VRM Core Code (CHASE 2026)

Core implementation of **Variational Risk Minimization (VRM)** for chest X-ray triage:
- multimodal teacher (BiomedCLIP + LQA),
- image-only student distilled from marginalized teacher targets,
- offline LVLM sampling for Monte Carlo report variants.

## What Is Included

```
github_upload/
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
