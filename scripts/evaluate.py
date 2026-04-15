"""
Unified Evaluation Script for VRM Models.

Evaluates both teacher and student models on test sets,
computing classification metrics (AUC, Accuracy, F1, etc.).

Usage:
    # Evaluate student (image-only)
    python evaluate.py \
        --model student \
        --checkpoint outputs/student_vrm/best_model.pt \
        --data_root data/mimic_cxr
    
    # Evaluate teacher (requires text)
    python evaluate.py \
        --model teacher \
        --checkpoint outputs/teacher/best_model.pt \
        --data_root data/mimic_cxr
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TeacherModel, EdgeStudent


class EvalDataset(Dataset):
    """Simple dataset for evaluation."""
    
    def __init__(self, data_root: str, split: str = "test", image_size: int = 224):
        self.data_root = Path(data_root)
        
        import pandas as pd
        labels_df = pd.read_csv(self.data_root / "labels.csv")
        
        with open(self.data_root / f"{split}_list.txt") as f:
            split_ids = set(line.strip() for line in f if line.strip())
        
        self.data = labels_df[labels_df['id'].isin(split_ids)].to_dict('records')
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _find_image(self, image_id: str):
        for ext in ['.png', '.jpg', '.jpeg']:
            path = self.data_root / "images" / f"{image_id}{ext}"
            if path.exists():
                return path
        return None
    
    def _load_report(self, image_id: str) -> str:
        for name in [f"{image_id}.txt", f"{image_id}_report.txt"]:
            path = self.data_root / "reports" / name
            if path.exists():
                return path.read_text().strip()
        return ""
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self._find_image(item['id'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        report = self._load_report(item['id'])
        
        return {
            'image': image,
            'report': report,
            'label': int(item['urgency_label']),
            'id': item['id']
        }


def compute_all_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute comprehensive classification metrics."""
    preds = (probs >= threshold).astype(int)
    
    metrics = {
        'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall': recall_score(labels, preds, average='macro', zero_division=0),
    }
    
    # Sensitivity & Specificity for binary classification
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return metrics


@torch.no_grad()
def evaluate_student(model: EdgeStudent, dataloader: DataLoader, device: torch.device) -> Dict:
    """Evaluate student model (image-only inference)."""
    model.eval()
    all_probs, all_labels = [], []
    
    for batch in tqdm(dataloader, desc="Evaluating Student"):
        images = batch['image'].to(device)
        labels = batch['label']
        
        out = model(images)
        probs = F.softmax(out['logits'], dim=1)[:, 1]
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    return compute_all_metrics(np.array(all_probs), np.array(all_labels))


@torch.no_grad()
def evaluate_teacher(model: TeacherModel, dataloader: DataLoader, device: torch.device) -> Dict:
    """Evaluate teacher model (requires text)."""
    model.eval()
    all_probs, all_labels = [], []
    
    for batch in tqdm(dataloader, desc="Evaluating Teacher"):
        images = batch['image'].to(device)
        reports = batch['report']
        labels = batch['label']
        
        # Tokenize
        encoded = model.tokenize(reports)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        out = model(images, input_ids, attention_mask)
        probs = F.softmax(out['logits'], dim=1)[:, 1]
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    return compute_all_metrics(np.array(all_probs), np.array(all_labels))


def main():
    parser = argparse.ArgumentParser(description='Evaluate VRM Models')
    parser.add_argument('--model', type=str, required=True, choices=['teacher', 'student'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--backbone', type=str, default='eva02_tiny_patch14_224', help='Student backbone')
    parser.add_argument(
        '--student_backbone_checkpoint',
        type=str,
        default='checkpoints/eva_x_tiny_patch16_merged520k_mim.pt',
        help='EVA-X checkpoint for student encoder transfer',
    )
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load dataset
    print(f"Loading {args.split} dataset from {args.data_root}...")
    dataset = EvalDataset(args.data_root, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Samples: {len(dataset)}")
    
    # Load model
    print(f"\nLoading {args.model} model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if args.model == 'student':
        model = EdgeStudent(
            backbone=args.backbone,
            pretrained=False,
            backbone_checkpoint=args.student_backbone_checkpoint,
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = evaluate_student(model, dataloader, device)
    else:
        model = TeacherModel().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = evaluate_teacher(model, dataloader, device)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.model.upper()} on {args.split})")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    
    results = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': len(dataset),
        'metrics': metrics,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
