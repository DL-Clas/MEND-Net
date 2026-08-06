import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TeacherImageEncoder(nn.Module):
    """ViT-B/32 based image encoder for teacher model.
    
    Uses pre-trained ViT-B/32 as described in the paper (approx 86M params).
    """
    def __init__(self, out_dim: int = 768, pretrained: bool = True):
        super().__init__()
        try:
            self.vit = models.vit_b_32(pretrained=pretrained)
        except Exception as exc:
            print(f"Warning: failed to load pretrained ViT-B/32 ({exc}); "
                  f"using random initialization.")
            self.vit = models.vit_b_32(weights=None)
        self.vit.heads = nn.Identity()
        self.proj = nn.Linear(768, out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.vit(x)
        return self.proj(feat)


class TeacherEHREncoder(nn.Module):
    """TabNet-inspired EHR encoder for teacher model.
    
    Processes structured clinical data with feature selection.
    """
    def __init__(self, in_dim: int = 12, out_dim: int = 64):
        super().__init__()
        self.shared_fc = nn.Linear(in_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, in_dim),
        )
        self.transform = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.shared_fc(x)))
        attention_weights = torch.softmax(self.attention(h), dim=-1)
        x_transformed = x * attention_weights
        return self.transform(x_transformed)


class TeacherClassifier(nn.Module):
    """Teacher model classification head with cross-modal fusion."""
    def __init__(self, img_dim: int = 768, ehr_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + ehr_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, img_feat: torch.Tensor, ehr_feat: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(torch.cat([img_feat, ehr_feat], dim=-1))
        return self.classifier(fused)


class TeacherModel(nn.Module):
    """Full teacher model: ViT-B/32 + TabNet-style EHR + Cross-modal Fusion.
    
    The paper reports a full-scale 312M teacher split as:
    - ViT-B/32: ~86M
    - TabNet EHR encoder: ~51M
    - Fusion + classifier: ~175M
    
    The released implementation reproduces the architecture exactly but uses
    lightweight MLP components for the TabNet-style EHR encoder and the
    fusion head (a few million parameters total). The dominant cost is the
    ViT-B/32 image encoder (~86M). To reproduce the exact 312M budget the
    fusion head should be widened; see the paper's Section 3.2.
    
    Args:
        num_classes: Number of disease classes.
        img_dim: Image feature dimension.
        ehr_dim: EHR feature dimension.
        ehr_input_dim: Raw EHR input dimension.
        pretrained: Whether to use pretrained ViT-B/32 weights.
    """
    def __init__(
        self,
        num_classes: int = 5,
        img_dim: int = 768,
        ehr_dim: int = 64,
        ehr_input_dim: int = 12,
        pretrained: bool = True,
    ):
        super().__init__()
        self.img_encoder = TeacherImageEncoder(out_dim=img_dim, pretrained=pretrained)
        self.ehr_encoder = TeacherEHREncoder(in_dim=ehr_input_dim, out_dim=ehr_dim)
        self.classifier = TeacherClassifier(img_dim, ehr_dim, num_classes)
    
    def forward(self, images: torch.Tensor, ehr: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: Chest X-ray images (B, 3, H, W).
            ehr: EHR features (B, 12).
            
        Returns:
            Classification logits (B, num_classes).
        """
        img_feat = self.img_encoder(images)
        ehr_feat = self.ehr_encoder(ehr)
        return self.classifier(img_feat, ehr_feat)
    
    def forward_with_features(self, images: torch.Tensor, ehr: torch.Tensor) -> dict:
        """Forward pass returning intermediate features for distillation."""
        img_feat = self.img_encoder(images)
        ehr_feat = self.ehr_encoder(ehr)
        logits = self.classifier(img_feat, ehr_feat)
        return {
            "logits": logits,
            "img_feat": img_feat,
            "ehr_feat": ehr_feat,
        }

