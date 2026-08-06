import torch
import torch.nn as nn
import torchvision.models as models

from .hd_nas import DifferentiableSearchSpace
from .hkd_module import HKDModule
from .dfc import DynamicFeatureCompensation


class ImageBackbone(nn.Module):
    """Lightweight image backbone using MobileNetV2 features.
    
    Extracts visual features from chest X-ray images.
    """
    def __init__(self, out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        try:
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
        except Exception as exc:
            print(f"Warning: failed to load pretrained MobileNetV2 ({exc}); "
                  f"using random initialization.")
            mobilenet = models.mobilenet_v2(weights=None)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, out_dim)
    
    def forward(self, x: torch.Tensor) -> tuple:
        feat_map = self.features(x)
        feat = self.pool(feat_map).flatten(1)
        out = self.fc(feat)
        return out, feat_map


class EHRBackbone(nn.Module):
    """MLP backbone for EHR (structured clinical data).
    
    Processes 12-dimensional EHR lab values.
    """
    def __init__(self, in_dim: int = 12, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierHead(nn.Module):
    """Classification head for 5-class disease diagnosis."""
    def __init__(self, in_dim: int = 192, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class HEALNet(nn.Module):
    """HEAL-Net: Hierarchical Knowledge Distillation and NAS Framework.
    
    Complete model integrating:
    1. HD-NAS: Hardware-aware Differentiable Architecture Search
    2. HKD: Hierarchical Knowledge Distillation
    3. DFC: Dynamic Feature Compensation
    
    Args:
        num_classes: Number of disease classes.
        img_dim: Image feature dimension.
        ehr_dim: EHR feature dimension.
        ehr_input_dim: Raw EHR input dimension.
        num_nodes: NAS search space nodes.
        lambda_flops: Hardware penalty weight.
        snr_threshold: DFC SNR threshold.
        gamma: HKD relation loss weight.
        beta: HKD KL loss weight.
        temperature: HKD distillation temperature.
    """
    def __init__(
        self,
        num_classes: int = 5,
        img_dim: int = 128,
        ehr_dim: int = 64,
        ehr_input_dim: int = 12,
        num_nodes: int = 4,
        lambda_flops: float = 0.05,
        snr_threshold: float = 0.3,
        gamma: float = 1.0,
        beta: float = 0.5,
        temperature: float = 3.0,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.img_backbone = ImageBackbone(out_dim=img_dim, pretrained=pretrained)
        self.ehr_backbone = EHRBackbone(in_dim=ehr_input_dim, out_dim=ehr_dim)
        
        self.search_space = DifferentiableSearchSpace(
            in_channels=32,
            hidden_channels=32,
            num_nodes=num_nodes,
        )
        
        self.img_proj = nn.Linear(img_dim, 32)
        self.fusion = nn.Sequential(
            nn.Linear(32 + ehr_dim, 128),
            nn.ReLU(inplace=True),
        )
        
        self.dfc = DynamicFeatureCompensation(
            img_dim=128,
            ehr_dim=ehr_dim,
            hidden_dim=128,
            snr_threshold=snr_threshold,
        )
        
        self.classifier = ClassifierHead(in_dim=128 + ehr_dim, num_classes=num_classes)
        
        self.hkd = HKDModule(
            teacher_img_dim=768,
            teacher_ehr_dim=64,
            student_img_dim=128,
            student_ehr_dim=ehr_dim,
            proj_dim=128,
            temperature=temperature,
            gamma=gamma,
            beta=beta,
        )
        
        self.nas_disabled = False
        self.hkd_disabled = False
        self.dfc_disabled = False
    
    def forward(
        self,
        images: torch.Tensor,
        ehr: torch.Tensor,
        ehr_mask: torch.Tensor = None,
        img_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for inference.
        
        Args:
            images: Chest X-ray images (B, 3, H, W).
            ehr: EHR features (B, 12).
            ehr_mask: Optional EHR availability mask (B,).
            img_mask: Optional image availability mask (B,).
            
        Returns:
            Classification logits (B, num_classes).
        """
        img_feat, feat_map = self.img_backbone(images)
        ehr_feat = self.ehr_backbone(ehr)
        
        feat_32 = self.img_proj(img_feat)
        B, C, H, W = feat_map.shape
        feat_32_spatial = feat_32.unsqueeze(-1).unsqueeze(-1).expand(B, 32, H, W)
        searched_feat = self.search_space(feat_32_spatial)
        searched_feat = searched_feat.mean(dim=[2, 3])
        
        fused = self.fusion(torch.cat([searched_feat, ehr_feat], dim=-1))
        
        dfc_out = self.dfc(fused, ehr_feat, img_mask=img_mask, ehr_mask=ehr_mask)
        comp_img = dfc_out["comp_img"]
        comp_ehr = dfc_out["comp_ehr"]
        
        combined = torch.cat([comp_img, comp_ehr], dim=-1)
        logits = self.classifier(combined)
        
        return logits
    
    def forward_with_features(
        self,
        images: torch.Tensor,
        ehr: torch.Tensor,
        ehr_mask: torch.Tensor = None,
        img_mask: torch.Tensor = None,
    ) -> dict:
        """Forward pass returning intermediate features for distillation.
        
        Returns:
            Dict with logits, img_feat, ehr_feat, and DFC outputs.
        """
        img_feat, feat_map = self.img_backbone(images)
        ehr_feat = self.ehr_backbone(ehr)
        
        feat_32 = self.img_proj(img_feat)
        B, C, H, W = feat_map.shape
        feat_32_spatial = feat_32.unsqueeze(-1).unsqueeze(-1).expand(B, 32, H, W)
        searched_feat = self.search_space(feat_32_spatial)
        searched_feat = searched_feat.mean(dim=[2, 3])
        
        fused = self.fusion(torch.cat([searched_feat, ehr_feat], dim=-1))
        
        dfc_out = self.dfc(fused, ehr_feat, img_mask=img_mask, ehr_mask=ehr_mask)
        comp_img = dfc_out["comp_img"]
        comp_ehr = dfc_out["comp_ehr"]
        
        combined = torch.cat([comp_img, comp_ehr], dim=-1)
        logits = self.classifier(combined)
        
        return {
            "logits": logits,
            "img_feat": comp_img,
            "ehr_feat": comp_ehr,
            "raw_img_feat": img_feat,
            "raw_ehr_feat": ehr_feat,
            "dfc_out": dfc_out,
        }
    
    def get_flops(self, input_size: int = 7) -> float:
        """Estimate total FLOPs.

        Returns the expected FLOPs of the searched fusion topology.
        For a full-network FLOPs estimate (including backbone, fusion,
        DFC and classifier) use the fvcore-based profiler in
        `utils.metrics.compute_flops`.
        """
        return self.search_space.get_flops(input_size=input_size)
