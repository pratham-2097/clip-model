"""
Hierarchical Binary Classifier Model

Frozen CLIP ViT-B/32 + Spatial Features + Object Type → Binary Classification

Architecture:
- CLIP ViT-B/32 (frozen): 512-d visual embedding
- Object type embedding (trainable): 32-d
- Spatial features: 9-d numerical
- MLP head (trainable): [553 → 256 → 128 → 2]

Only ~150K parameters trained (0.1% of total).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class HierarchicalBinaryClassifier(nn.Module):
    """
    Hierarchical binary classifier using frozen CLIP + spatial reasoning.
    
    Components:
    - CLIP ViT-B/32 (frozen): Extracts visual features
    - Object type embedding (trainable): Encodes object type info
    - Spatial features: Numerical features from YOLO detections
    - MLP head (trainable): Combines all features for binary classification
    """
    
    def __init__(
        self,
        clip_model_name: str = 'openai/clip-vit-base-patch32',
        num_object_types: int = 4,
        object_type_embed_dim: int = 32,
        spatial_feature_dim: int = 9,
        hidden_dims: list = [256, 128],
        dropout: float = 0.3,
    ):
        """
        Args:
            clip_model_name: CLIP model to use
            num_object_types: Number of object types (4: rock_toe, slope_drain, toe_drain, vegetation)
            object_type_embed_dim: Dimension of object type embedding
            spatial_feature_dim: Number of spatial features (9)
            hidden_dims: Hidden layer dimensions for MLP
            dropout: Dropout probability
        """
        super().__init__()
        
        # Frozen CLIP model
        print(f"Loading CLIP model: {clip_model_name}...")
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze all CLIP parameters
        for param in self.clip.parameters():
            param.requires_grad = False
        print("✅ CLIP backbone frozen (vision + text encoders)")
        
        # Get CLIP vision embedding dimension
        self.clip_embed_dim = self.clip.config.vision_config.hidden_size  # 768 for ViT-B/32
        self.clip_projection_dim = self.clip.config.projection_dim  # 512 for ViT-B/32
        
        # Object type embedding (trainable)
        self.object_type_embedding = nn.Embedding(num_object_types, object_type_embed_dim)
        
        # MLP head (trainable)
        input_dim = self.clip_projection_dim + object_type_embed_dim + spatial_feature_dim
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer (2 classes: NORMAL, CONDITIONAL)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize embeddings and MLP
        self._init_weights()
    
    def _init_weights(self):
        """Initialize trainable parameters."""
        # Initialize object type embeddings
        nn.init.normal_(self.object_type_embedding.weight, std=0.02)
        
        # Initialize MLP layers
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        object_type_ids: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pixel_values: [B, 3, 224, 224] - CLIP-preprocessed images
            object_type_ids: [B] - Object type indices
            spatial_features: [B, 9] - Spatial features
        
        Returns:
            logits: [B, 2] - Binary classification logits
        """
        batch_size = pixel_values.shape[0]
        
        # Get CLIP visual embeddings (frozen, no gradient)
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            # Get pooled output (CLS token)
            visual_embeds = vision_outputs.pooler_output  # [B, 768]
            # Project to CLIP's projection space
            visual_embeds = self.clip.visual_projection(visual_embeds)  # [B, 512]
            # Normalize (as CLIP does)
            visual_embeds = visual_embeds / visual_embeds.norm(dim=-1, keepdim=True)
        
        # Get object type embeddings (trainable)
        type_embeds = self.object_type_embedding(object_type_ids)  # [B, 32]
        
        # Concatenate all features
        combined = torch.cat([visual_embeds, type_embeds, spatial_features], dim=1)
        # Shape: [B, 512 + 32 + 9] = [B, 553]
        
        # MLP head (trainable)
        logits = self.mlp(combined)  # [B, 2]
        
        return logits
    
    def get_trainable_params(self):
        """Get trainable parameters and count."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable_params': trainable_params,
            'trainable_count': trainable_count,
            'total_count': total_count,
            'trainable_percentage': 100 * trainable_count / total_count,
        }
    
    def print_parameter_summary(self):
        """Print summary of model parameters."""
        info = self.get_trainable_params()
        
        print(f"\n{'='*60}")
        print("Model Parameter Summary")
        print(f"{'='*60}")
        print(f"Total parameters:      {info['total_count']:,}")
        print(f"Trainable parameters:  {info['trainable_count']:,}")
        print(f"Frozen parameters:     {info['total_count'] - info['trainable_count']:,}")
        print(f"Trainable percentage:  {info['trainable_percentage']:.2f}%")
        print(f"{'='*60}\n")


def test_model():
    """Test the model with dummy data."""
    print("Testing HierarchicalBinaryClassifier...")
    
    # Create model
    model = HierarchicalBinaryClassifier(
        clip_model_name='openai/clip-vit-base-patch32',
        num_object_types=4,
        object_type_embed_dim=32,
        spatial_feature_dim=9,
        hidden_dims=[256, 128],
        dropout=0.3,
    )
    
    # Print parameter summary
    model.print_parameter_summary()
    
    # Test forward pass
    batch_size = 4
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    object_type_ids = torch.randint(0, 4, (batch_size,))
    spatial_features = torch.rand(batch_size, 9)
    
    print("Testing forward pass...")
    print(f"  Input pixel_values shape: {pixel_values.shape}")
    print(f"  Input object_type_ids shape: {object_type_ids.shape}")
    print(f"  Input spatial_features shape: {spatial_features.shape}")
    
    with torch.no_grad():
        logits = model(pixel_values, object_type_ids, spatial_features)
    
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Sample logits: {logits[0]}")
    
    # Test that gradients only flow through trainable parts
    print("\nTesting gradient flow...")
    logits = model(pixel_values, object_type_ids, spatial_features)
    loss = logits.sum()
    loss.backward()
    
    # Check CLIP parameters have no gradients
    clip_has_grad = any(p.grad is not None for p in model.clip.parameters())
    mlp_has_grad = any(p.grad is not None for p in model.mlp.parameters())
    embed_has_grad = model.object_type_embedding.weight.grad is not None
    
    print(f"  CLIP has gradients: {clip_has_grad} (should be False)")
    print(f"  MLP has gradients: {mlp_has_grad} (should be True)")
    print(f"  Object embedding has gradients: {embed_has_grad} (should be True)")
    
    assert not clip_has_grad, "CLIP should not have gradients!"
    assert mlp_has_grad, "MLP should have gradients!"
    assert embed_has_grad, "Object embedding should have gradients!"
    
    print("\n✅ Model working correctly!")
    print("✅ CLIP frozen (no gradients)")
    print("✅ MLP and embeddings trainable (has gradients)")
    print("✅ Only ~150K parameters will be trained (0.1% of total)")


if __name__ == '__main__':
    test_model()

