"""
VRM Student Distillation Training Script.

Trains the lightweight EVA-based student model via VRM distillation
from a frozen multimodal teacher model.

Key components:
1. Load K variational samples per image (generated offline)
2. Teacher processes each sample -> marginalize predictions
3. Student learns from marginalized soft targets + feature alignment

Usage:
    python train_student_vrm.py \
        --data_root data/mimic_cxr \
        --teacher_checkpoint outputs/teacher/best_model.pt \
        --samples_file train_variational_samples.json \
        --output_dir outputs/student_vrm
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TeacherModel, EdgeStudent, VRMLoss


class VRMDataset(Dataset):
    """Dataset with variational samples for VRM training."""
    
    def __init__(
        self,
        data_root: str,
        samples_file: str,
        split: str = "train",
        k_samples: int = 5,
        image_size: int = 224,
    ):
        self.data_root = Path(data_root)
        self.k_samples = k_samples
        
        # Load labels
        import pandas as pd
        labels_df = pd.read_csv(self.data_root / "labels.csv")
        
        # Load split
        with open(self.data_root / f"{split}_list.txt") as f:
            split_ids = set(line.strip() for line in f if line.strip())
        
        self.data = labels_df[labels_df['id'].isin(split_ids)].to_dict('records')
        
        # Load variational samples
        with open(samples_file) as f:
            self.variational_samples = json.load(f)
        
        # Transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def _find_image(self, image_id: str) -> Optional[Path]:
        for ext in ['.png', '.jpg', '.jpeg']:
            path = self.data_root / "images" / f"{image_id}{ext}"
            if path.exists():
                return path
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item['id']
        
        # Load image
        image_path = self._find_image(image_id)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Get variational samples
        samples = self.variational_samples.get(image_id, ["No report available."])
        samples = samples[:self.k_samples]
        # Pad if less than k_samples
        while len(samples) < self.k_samples:
            samples.append(samples[0])
        
        return {
            'image': image,
            'variational_reports': samples,
            'label': int(item['urgency_label']),
            'id': image_id
        }


class FeatureProjector(nn.Module):
    """Projects student features to match teacher dimension."""
    
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, teacher_dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.proj(x)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute classification metrics."""
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    
    probs_np = probs[:, 1].cpu().numpy()
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = 0.5
    
    return {
        'accuracy': accuracy_score(labels_np, preds_np),
        'f1': f1_score(labels_np, preds_np, average='macro'),
        'auc': auc,
    }


def train_epoch_vrm(
    student: EdgeStudent,
    teacher: TeacherModel,
    projector: FeatureProjector,
    dataloader: DataLoader,
    criterion: VRMLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    k_samples: int = 5,
) -> Dict[str, float]:
    """
    Train one epoch using VRM distillation.
    
    For each batch:
    1. Teacher processes K variational samples -> K predictions
    2. Marginalize teacher predictions (average)
    3. Student predicts from image only
    4. Compute VRM loss (KL + feature + classification)
    """
    student.train()
    teacher.eval()
    projector.train()
    
    total_loss = 0
    all_logits, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="VRM Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        variational_reports = batch['variational_reports']  # List of K reports per sample
        batch_size = images.shape[0]
        
        # Gather teacher predictions for K samples
        teacher_logits_list = []
        teacher_features_list = []
        
        with torch.no_grad():
            for k in range(k_samples):
                # Get k-th report for each sample
                reports_k = [variational_reports[k][i] for i in range(batch_size)]
                
                # Tokenize
                encoded = teacher.tokenize(reports_k)
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                # Teacher forward
                teacher_out = teacher(images, input_ids, attention_mask)
                teacher_logits_list.append(teacher_out['logits'])
                teacher_features_list.append(teacher_out['fused_features'].mean(dim=1))
        
        # Marginalize: average over K samples
        teacher_logits_stack = torch.stack(teacher_logits_list, dim=0)  # [K, B, C]
        teacher_logits_marginalized = teacher_logits_stack.mean(dim=0)  # [B, C]
        
        teacher_features_stack = torch.stack(teacher_features_list, dim=0)  # [K, B, D]
        teacher_features_marginalized = teacher_features_stack.mean(dim=0)  # [B, D]
        
        # Student forward (image only - no text at inference!)
        student_out = student(images, return_features=True)
        student_logits = student_out['logits']
        student_features = projector(student_out['features'])
        
        # VRM Loss
        loss_dict = criterion(
            student_logits=student_logits,
            teacher_logits_marginalized=teacher_logits_marginalized,
            student_features=student_features,
            teacher_features=teacher_features_marginalized,
            labels=labels,
        )
        loss = loss_dict['total_loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(projector.parameters()), 1.0
        )
        optimizer.step()
        
        total_loss += loss.item()
        all_logits.append(student_logits.detach())
        all_labels.append(labels.detach())
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'kl': f"{loss_dict['kl_loss'].item():.4f}",
        })
    
    # Epoch metrics
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


@torch.no_grad()
def evaluate(student: EdgeStudent, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate student model (image-only inference)."""
    student.eval()
    all_logits, all_labels = [], []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        out = student(images)
        all_logits.append(out['logits'])
        all_labels.append(labels)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    return compute_metrics(all_logits, all_labels)


def main():
    parser = argparse.ArgumentParser(description='VRM Student Distillation Training')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--samples_file', type=str, required=True, help='Variational samples JSON')
    parser.add_argument('--output_dir', type=str, default='outputs/student_vrm')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--k_samples', type=int, default=5)
    parser.add_argument('--lambda_kl', type=float, default=1.0)
    parser.add_argument('--lambda_feat', type=float, default=0.5)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--backbone', type=str, default='eva02_tiny_patch14_224')
    parser.add_argument(
        '--student_backbone_checkpoint',
        type=str,
        default='checkpoints/eva_x_tiny_patch16_merged520k_mim.pt',
        help='EVA-X checkpoint used for student encoder transfer',
    )
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    device = torch.device(args.device)
    
    # Load teacher (frozen)
    print("Loading teacher model...")
    teacher = TeacherModel().to(device)
    teacher.load_state_dict(torch.load(args.teacher_checkpoint)['model_state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("Teacher loaded and frozen")
    
    # Create student
    print(f"Creating student model ({args.backbone})...")
    student = EdgeStudent(
        backbone=args.backbone,
        pretrained=False,
        backbone_checkpoint=args.student_backbone_checkpoint,
    ).to(device)
    student_params = student.count_parameters()
    print(f"Student params: {student_params['total']:,}")
    print(f"Student backbone: {student.backbone_name}")
    print(f"Backbone init: {student.backbone_load_info}")
    
    # Feature projector (student -> teacher dimension)
    projector = FeatureProjector(
        student_dim=student.feature_dim,
        teacher_dim=teacher.hidden_dim
    ).to(device)
    
    # Datasets
    print("Loading datasets...")
    train_dataset = VRMDataset(args.data_root, args.samples_file, 'train', args.k_samples)
    val_dataset = VRMDataset(args.data_root, args.samples_file, 'val', args.k_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Loss and optimizer
    criterion = VRMLoss(
        lambda_kl=args.lambda_kl,
        lambda_feat=args.lambda_feat,
        lambda_cls=args.lambda_cls,
    )
    
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    print(f"\nStarting VRM distillation for {args.epochs} epochs...")
    best_auc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_metrics = train_epoch_vrm(
            student, teacher, projector, train_loader,
            criterion, optimizer, device, args.k_samples
        )
        scheduler.step()
        
        val_metrics = evaluate(student, val_loader, device)
        
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"  Val   - AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        
        # Save best
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'auc': best_auc,
            }, output_dir / 'best_model.pt')
            print(f"  Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    torch.save({'model_state_dict': student.state_dict()}, output_dir / 'final_model.pt')
    writer.close()
    print(f"\nTraining complete! Best AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
