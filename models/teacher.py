"""
Teacher Model with Latent Query Alignment (LQA) for VRM Framework.

Architecture:
- Vision Encoder: BiomedCLIP (frozen, last 3 layers unfrozen for fine-tuning)
- Text Encoder: BiomedCLIP text encoder (frozen, last 3 layers unfrozen)
- Fusion: Latent Query Alignment (LQA) - Q-Former style cross-modal fusion
- Classifier: MLP head for urgency classification

The LQA module bridges vision and language through learnable queries that
attend to both modalities, enabling effective multimodal reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from transformers import AutoModel, AutoTokenizer


class LatentQueryAlignment(nn.Module):
    """
    Latent Query Alignment (LQA) module for vision-language fusion.
    
    Inspired by Q-Former (BLIP-2), uses learnable queries to bridge
    the modality gap between image and text representations.
    
    Args:
        num_queries: Number of learnable query tokens
        hidden_dim: Hidden dimension for queries and attention
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_queries: int = 8,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Learnable query embeddings
        self.queries = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Self-attention layers (queries interact with text)
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # Cross-attention to image features
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.ln_self = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ln_cross = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ln_ffn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through LQA layers.
        
        Args:
            image_features: [B, num_patches, hidden_dim] from vision encoder
            text_features: [B, seq_len, hidden_dim] from text encoder (optional)
            
        Returns:
            queries: [B, num_queries, hidden_dim] fused query representations
        """
        batch_size = image_features.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)

        for i in range(len(self.self_attn_layers)):
            # Self-attention with optional text conditioning
            if text_features is not None:
                # Concat queries and text for joint self-attention
                combined = torch.cat([queries, text_features], dim=1)
                attn_out, _ = self.self_attn_layers[i](combined, combined, combined)
                queries = self.ln_self[i](queries + attn_out[:, :self.num_queries])
            else:
                attn_out, _ = self.self_attn_layers[i](queries, queries, queries)
                queries = self.ln_self[i](queries + attn_out)

            # Cross-attention to image features
            cross_out, _ = self.cross_attn_layers[i](queries, image_features, image_features)
            queries = self.ln_cross[i](queries + cross_out)

            # Feed-forward
            ffn_out = self.ffn_layers[i](queries)
            queries = self.ln_ffn[i](queries + ffn_out)

        return queries


class TeacherModel(nn.Module):
    """
    Teacher Model for VRM Framework.
    
    Combines BiomedCLIP vision and text encoders with LQA fusion
    for multimodal urgency classification.
    
    Args:
        model_name: HuggingFace model name for BiomedCLIP
        num_queries: Number of LQA queries
        num_heads: Number of attention heads in LQA
        num_layers: Number of LQA layers  
        dropout: Dropout rate
        num_classes: Number of output classes
        unfreeze_layers: Number of encoder layers to unfreeze for fine-tuning
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        num_queries: int = 8,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
        unfreeze_layers: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Load BiomedCLIP
        self.clip_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get hidden dimension from model config
        self.hidden_dim = self.clip_model.config.projection_dim
        
        # Freeze most layers, unfreeze last N for fine-tuning
        self._freeze_encoders(unfreeze_layers)
        
        # LQA fusion module (trainable)
        self.lqa = LatentQueryAlignment(
            num_queries=num_queries,
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Projection layers to match dimensions if needed
        vision_dim = self.clip_model.config.vision_config.hidden_size
        text_dim = self.clip_model.config.text_config.hidden_size
        
        self.vision_proj = nn.Linear(vision_dim, self.hidden_dim) if vision_dim != self.hidden_dim else nn.Identity()
        self.text_proj = nn.Linear(text_dim, self.hidden_dim) if text_dim != self.hidden_dim else nn.Identity()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
        
        self._init_weights()

    def _freeze_encoders(self, unfreeze_layers: int):
        """Freeze encoder layers except the last N."""
        # Freeze all parameters first
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last N layers of vision encoder
        vision_layers = self.clip_model.vision_model.encoder.layers
        for layer in vision_layers[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        # Unfreeze last N layers of text encoder
        text_layers = self.clip_model.text_model.encoder.layer
        for layer in text_layers[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def _init_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features using vision encoder."""
        outputs = self.clip_model.vision_model(images)
        features = outputs.last_hidden_state  # [B, num_patches+1, vision_dim]
        return self.vision_proj(features)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract text features using text encoder."""
        outputs = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state  # [B, seq_len, text_dim]
        return self.text_proj(features)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through teacher model.
        
        Args:
            images: [B, 3, H, W] input images
            input_ids: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask
            
        Returns:
            Dictionary containing:
                - logits: [B, num_classes] classification logits
                - fused_features: [B, num_queries, hidden_dim] LQA output
                - image_features: [B, num_patches, hidden_dim] vision features
                - text_features: [B, seq_len, hidden_dim] text features
        """
        # Encode both modalities
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)
        
        # Fuse with LQA
        fused_features = self.lqa(image_features, text_features)
        
        # Classification from mean-pooled queries
        pooled = fused_features.mean(dim=1)  # [B, hidden_dim]
        logits = self.classifier(pooled)
        
        return {
            "logits": logits,
            "fused_features": fused_features,
            "image_features": image_features,
            "text_features": text_features,
        }

    def tokenize(self, texts: list, max_length: int = 256) -> Dict[str, torch.Tensor]:
        """Tokenize input texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )


def create_teacher_model(
    model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    num_queries: int = 8,
    num_heads: int = 8,
    num_layers: int = 3,
    dropout: float = 0.2,
    num_classes: int = 2,
    device: str = "cuda",
) -> TeacherModel:
    """
    Factory function to create and initialize teacher model.
    
    Args:
        model_name: BiomedCLIP model identifier
        num_queries: Number of LQA queries
        num_heads: Number of attention heads
        num_layers: Number of LQA layers
        dropout: Dropout rate
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        Initialized TeacherModel
    """
    model = TeacherModel(
        model_name=model_name,
        num_queries=num_queries,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
    )
    return model.to(device)


if __name__ == "__main__":
    # Quick test
    model = create_teacher_model(device="cpu")
    print(f"Teacher model created successfully")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
