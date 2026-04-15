"""
Loss Functions for VRM Framework.

Implements the three-component VRM loss:
1. KL Divergence Loss - Aligns student with marginalized teacher predictions
2. Feature Distillation Loss - Aligns student features with teacher embeddings
3. Classification Loss - Standard cross-entropy with ground truth labels

Total Loss: L_VRM = λ_KL * L_KL + λ_feat * L_feat + λ_cls * L_cls
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for image-text alignment.
    Used during teacher training for cross-modal learning.
    
    Args:
        temperature: Softmax temperature for similarity scaling
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bidirectional contrastive loss.
        
        Args:
            image_features: [B, D] normalized image embeddings
            text_features: [B, D] normalized text embeddings
            
        Returns:
            Contrastive loss scalar
        """
        # L2 normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Bidirectional cross-entropy
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2


class ClassificationLoss(nn.Module):
    """
    Classification loss with optional focal loss for class imbalance.
    
    Args:
        num_classes: Number of output classes
        label_smoothing: Label smoothing factor
        focal_gamma: Focal loss gamma (0 = standard CE)
        class_weights: Optional per-class weights
    """

    def __init__(
        self,
        num_classes: int = 2,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        
        if focal_gamma > 0:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=class_weights
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            logits: [B, num_classes] model predictions
            targets: [B] ground truth labels
            
        Returns:
            Classification loss scalar
        """
        if self.focal_gamma > 0:
            # Focal loss
            ce_loss = self.ce_loss(logits, targets)
            probs = F.softmax(logits, dim=1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.focal_gamma
            return (focal_weight * ce_loss).mean()
        else:
            return self.ce_loss(logits, targets)


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation loss.
    Aligns student features with teacher's fused representations.
    
    Args:
        temperature: Temperature for feature scaling
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature alignment loss using cosine distance.
        
        Args:
            student_features: [B, D] student feature vectors
            teacher_features: [B, D] teacher feature vectors (detached)
            
        Returns:
            Feature distillation loss scalar
        """
        # L2 normalize both
        student_norm = F.normalize(student_features, dim=-1)
        teacher_norm = F.normalize(teacher_features.detach(), dim=-1)
        
        # Cosine similarity -> distance
        similarity = (student_norm * teacher_norm).sum(dim=-1)
        distance = 1 - similarity
        
        return distance.mean() / self.temperature


class VRMLoss(nn.Module):
    """
    Marginalized Risk Minimization Loss.
    
    Combines three components:
    1. KL divergence between student and marginalized teacher predictions
    2. Feature distillation loss for representation alignment
    3. Classification loss with ground truth labels
    
    Args:
        lambda_kl: Weight for KL divergence loss
        lambda_feat: Weight for feature distillation loss
        lambda_cls: Weight for classification loss
        temperature: Softmax temperature for KL computation
        feature_temperature: Temperature for feature distillation
        num_classes: Number of output classes
    """

    def __init__(
        self,
        lambda_kl: float = 1.0,
        lambda_feat: float = 0.5,
        lambda_cls: float = 1.0,
        temperature: float = 2.0,
        feature_temperature: float = 4.0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_feat = lambda_feat
        self.lambda_cls = lambda_cls
        self.temperature = temperature
        
        self.feature_loss = FeatureDistillationLoss(temperature=feature_temperature)
        self.classification_loss = ClassificationLoss(num_classes=num_classes)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits_marginalized: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VRM distillation loss.
        
        Args:
            student_logits: [B, C] student predictions
            teacher_logits_marginalized: [B, C] averaged teacher predictions over K samples
            student_features: [B, D] student features (optional)
            teacher_features: [B, D] averaged teacher features (optional)
            labels: [B] ground truth labels (optional)
            
        Returns:
            Dictionary with total_loss and component losses
        """
        losses = {}
        total_loss = 0.0
        
        # 1. KL Divergence Loss
        # Soft targets from marginalized teacher predictions
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits_marginalized / self.temperature, dim=1)
        
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)  # Scale by T^2 as per Hinton et al.
        
        losses['kl_loss'] = kl_loss
        total_loss += self.lambda_kl * kl_loss
        
        # 2. Feature Distillation Loss
        if student_features is not None and teacher_features is not None:
            feat_loss = self.feature_loss(student_features, teacher_features)
            losses['feature_loss'] = feat_loss
            total_loss += self.lambda_feat * feat_loss
        else:
            losses['feature_loss'] = torch.tensor(0.0, device=student_logits.device)
        
        # 3. Classification Loss (with ground truth)
        if labels is not None:
            cls_loss = self.classification_loss(student_logits, labels.long())
            losses['classification_loss'] = cls_loss
            total_loss += self.lambda_cls * cls_loss
        else:
            losses['classification_loss'] = torch.tensor(0.0, device=student_logits.device)
        
        losses['total_loss'] = total_loss
        
        return losses


class TeacherLoss(nn.Module):
    """
    Loss function for teacher model training.
    
    Combines contrastive learning and classification:
    L_teacher = λ_contrast * L_contrast + λ_cls * L_cls
    
    Args:
        lambda_contrast: Weight for contrastive loss
        lambda_cls: Weight for classification loss
        temperature: Contrastive loss temperature
        num_classes: Number of output classes
    """

    def __init__(
        self,
        lambda_contrast: float = 0.3,
        lambda_cls: float = 0.7,
        temperature: float = 0.07,
        num_classes: int = 2,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_cls = lambda_cls
        
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.classification_loss = ClassificationLoss(
            num_classes=num_classes,
            label_smoothing=label_smoothing
        )

    def forward(
        self,
        fused_features: torch.Tensor,
        text_features: torch.Tensor,
        classification_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute teacher training loss.
        
        Args:
            fused_features: [B, num_queries, D] LQA output features
            text_features: [B, seq_len, D] text encoder features
            classification_logits: [B, num_classes] classifier output
            labels: [B] ground truth labels
            
        Returns:
            Dictionary with total_loss and component losses
        """
        # Pool features to [B, D]
        img_embed = fused_features.mean(dim=1)
        text_embed = text_features.mean(dim=1)
        
        # Contrastive loss
        contrast_loss = self.contrastive_loss(img_embed, text_embed)
        
        # Classification loss
        cls_loss = self.classification_loss(classification_logits, labels.long())
        
        total_loss = self.lambda_contrast * contrast_loss + self.lambda_cls * cls_loss
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrast_loss,
            'classification_loss': cls_loss,
        }


if __name__ == "__main__":
    # Quick test
    batch_size, num_classes, feature_dim = 4, 2, 768
    
    # VRM Loss test
    vrm_loss = VRMLoss()
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    student_feat = torch.randn(batch_size, feature_dim)
    teacher_feat = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    losses = vrm_loss(student_logits, teacher_logits, student_feat, teacher_feat, labels)
    print("VRM Loss components:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
