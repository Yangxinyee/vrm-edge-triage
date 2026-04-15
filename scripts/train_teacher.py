"""
Teacher Model Training Script.

Trains the BiomedCLIP-based teacher model with LQA fusion
for multimodal urgency classification.

Usage:
    python train_teacher.py \
        --data_root data/mimic_cxr \
        --output_dir outputs/teacher \
        --epochs 20 \
        --batch_size 32
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from PIL import Image
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import TeacherModel, TeacherLoss


class TriageDataset(Dataset):
    """Simple dataset for teacher training."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 224,
    ):
        self.data_root = Path(data_root)
        self.split = split
        
        # Load labels
        labels_df = self._load_labels()
        
        # Load split list
        split_file = self.data_root / f"{split}_list.txt"
        with open(split_file) as f:
            split_ids = set(line.strip() for line in f if line.strip())
        
        # Filter by split
        self.data = labels_df[labels_df['id'].isin(split_ids)].to_dict('records')
        
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
    
    def _load_labels(self):
        import pandas as pd
        return pd.read_csv(self.data_root / "labels.csv")
    
    def _find_image(self, image_id: str) -> Optional[Path]:
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
        
        # Load image
        image_path = self._find_image(item['id'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Load report
        report = self._load_report(item['id'])
        
        return {
            'image': image,
            'report': report,
            'label': int(item['urgency_label']),
            'id': item['id']
        }


def collate_fn(batch, tokenizer, max_length=256):
    """Custom collate function with text tokenization."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    reports = [item['report'] for item in batch]
    
    # Tokenize reports
    encoded = tokenizer(
        reports,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        'images': images,
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': labels,
    }


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute classification metrics."""
    probs = torch.softmax(logits, dim=1)
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_logits, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast('cuda'):
                outputs = model(images, input_ids, attention_mask)
                loss_dict = criterion(
                    outputs['fused_features'],
                    outputs['text_features'],
                    outputs['logits'],
                    labels
                )
                loss = loss_dict['total_loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, input_ids, attention_mask)
            loss_dict = criterion(
                outputs['fused_features'],
                outputs['text_features'],
                outputs['logits'],
                labels
            )
            loss = loss_dict['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        all_logits.append(outputs['logits'].detach())
        all_labels.append(labels.detach())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute epoch metrics
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    all_logits, all_labels = [], []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(images, input_ids, attention_mask)
        
        all_logits.append(outputs['logits'])
        all_labels.append(labels)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    return compute_metrics(all_logits, all_labels)


def main():
    parser = argparse.ArgumentParser(description='Train VRM Teacher Model')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/teacher')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_queries', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    device = torch.device(args.device)
    
    # Create model
    print("Creating teacher model...")
    model = TeacherModel(
        num_queries=args.num_queries,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = TriageDataset(args.data_root, split='train')
    val_dataset = TriageDataset(args.data_root, split='val')
    
    # Create dataloaders with tokenizer
    from functools import partial
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=partial(collate_fn, tokenizer=model.tokenizer),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=partial(collate_fn, tokenizer=model.tokenizer),
        pin_memory=True,
    )
    
    # Loss and optimizer
    criterion = TeacherLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda') if args.use_amp else None
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_auc = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"  Val   - AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        
        # Save best model
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': best_auc,
            }, output_dir / 'best_model.pt')
            print(f"  Saved best model (AUC: {best_auc:.4f})")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
    }, output_dir / 'final_model.pt')
    
    writer.close()
    print(f"\nTraining complete! Best AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
