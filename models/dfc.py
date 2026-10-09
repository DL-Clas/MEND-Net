import torch
import torch.nn as nn
import torch.nn.functional as F


class SNRMaskGenerator(nn.Module):
    """Compute SNR-based binary mask for modality availability.
    
    Implements Eq.15-16 from MEND-Net paper:
    M_m = I(SNR(X_m) > tau)
    
    Args:
        threshold: SNR threshold for mask generation.
    """
    def __init__(self, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold
    
    def compute_snr(self, x: torch.Tensor) -> torch.Tensor:
        """Compute signal-to-noise ratio estimate.
        
        Args:
            x: Input tensor (B, C, H, W) or (B, D).
            
        Returns:
            SNR values (B,).
        """
        if x.dim() == 4:
            signal_power = x.mean(dim=[1, 2, 3]) ** 2
            noise_power = x.var(dim=[1, 2, 3])
        else:
            signal_power = x.mean(dim=-1) ** 2
            noise_power = x.var(dim=-1)
        
        snr = signal_power / (noise_power + 1e-8)
        return snr
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate binary mask based on SNR threshold.
        
        Args:
            x: Input features.
            
        Returns:
            Binary mask (B,) with 1=available, 0=missing.
        """
        snr = self.compute_snr(x)
        mask = (snr > self.threshold).float()
        return mask


class CrossModalGenerator(nn.Module):
    """Lightweight cross-modal feature generator.
    
    Implements Eq.17-18 from MEND-Net paper:
    Z_c = MLP_c([F_img || F_ehr || M])
    F_bar_m = G_theta(Z_c) * (1 - M_m)
    
    Uses MLP + cross-attention for feature reconstruction.
    
    Args:
        input_dim: Total input dimension (img_dim + ehr_dim + mask_dim).
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension (same as missing modality).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        
        self.context_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        observed_features: list,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate reconstructed features for missing modality.
        
        Args:
            observed_features: List of available modality features.
            mask: Concatenated mask vector (B, mask_dim).
            
        Returns:
            Reconstructed features (B, output_dim).
        """
        concatenated = torch.cat([*observed_features, mask], dim=-1)
        context = self.context_mlp(concatenated)
        
        context_seq = context.unsqueeze(1)
        attn_out, _ = self.cross_attn(context_seq, context_seq, context_seq)
        context = self.norm(context + attn_out.squeeze(1))
        
        return self.generator(context)


class GatedFusion(nn.Module):
    """Adaptive gating fusion for observed and reconstructed features.
    
    Implements Eq.20-21 from MEND-Net paper:
    g = sigma(W_g * (F_hat + F_bar) + b_g)
    F_comp = g * F_hat + (1 - g) * F_bar
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        observed: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse observed and reconstructed features via learned gate.
        
        Args:
            observed: Observed (possibly masked) features (B, D).
            reconstructed: Reconstructed features (B, D).
            
        Returns:
            Compensated features (B, D).
        """
        gate_input = torch.cat([observed, reconstructed], dim=-1)
        g = self.gate(gate_input)
        return g * observed + (1 - g) * reconstructed


class DynamicFeatureCompensation(nn.Module):
    """Dynamic Feature Compensation (DFC) module.
    
    Handles modality missing scenarios at inference time:
    1. Compute SNR-based masks
    2. Generate reconstructed features for missing modalities
    3. Fuse via adaptive gating
    
    Args:
        img_dim: Image feature dimension.
        ehr_dim: EHR feature dimension.
        hidden_dim: Hidden dimension for generator.
        snr_threshold: SNR threshold for mask generation.
    """
    def __init__(
        self,
        img_dim: int = 128,
        ehr_dim: int = 64,
        hidden_dim: int = 128,
        snr_threshold: float = 0.3,
    ):
        super().__init__()
        self.img_dim = img_dim
        self.ehr_dim = ehr_dim
        
        self.img_mask_gen = SNRMaskGenerator(threshold=snr_threshold)
        self.ehr_mask_gen = SNRMaskGenerator(threshold=snr_threshold)
        
        mask_dim = 2
        self.img_generator = CrossModalGenerator(
            input_dim=ehr_dim + mask_dim,
            hidden_dim=hidden_dim,
            output_dim=img_dim,
        )
        self.ehr_generator = CrossModalGenerator(
            input_dim=img_dim + mask_dim,
            hidden_dim=hidden_dim,
            output_dim=ehr_dim,
        )
        
        self.img_fusion = GatedFusion(img_dim)
        self.ehr_fusion = GatedFusion(ehr_dim)
    
    def forward(
        self,
        img_feat: torch.Tensor,
        ehr_feat: torch.Tensor,
        img_mask: torch.Tensor = None,
        ehr_mask: torch.Tensor = None,
    ) -> dict:
        """Apply dynamic feature compensation.
        
        Args:
            img_feat: Image features (B, D_img).
            ehr_feat: EHR features (B, D_ehr).
            img_mask: Optional pre-computed image mask (B,).
            ehr_mask: Optional pre-computed EHR mask (B,).
            
        Returns:
            Dict with compensated features and masks.
        """
        B = img_feat.shape[0]
        device = img_feat.device
        
        if img_mask is None:
            img_mask = self.img_mask_gen(img_feat)
        if ehr_mask is None:
            ehr_mask = self.ehr_mask_gen(ehr_feat)
        
        img_mask = img_mask.to(device)
        ehr_mask = ehr_mask.to(device)
        
        masked_img = img_feat * img_mask.unsqueeze(-1)
        masked_ehr = ehr_feat * ehr_mask.unsqueeze(-1)
        
        mask_vec = torch.stack([img_mask, ehr_mask], dim=-1)
        
        img_recon = torch.zeros_like(img_feat)
        ehr_recon = torch.zeros_like(ehr_feat)
        
        missing_img = (img_mask == 0)
        if missing_img.any():
            recon = self.img_generator(
                [masked_ehr[missing_img]],
                mask_vec[missing_img],
            )
            img_recon[missing_img] = recon
        
        missing_ehr = (ehr_mask == 0)
        if missing_ehr.any():
            recon = self.ehr_generator(
                [masked_img[missing_ehr]],
                mask_vec[missing_ehr],
            )
            ehr_recon[missing_ehr] = recon
        
        comp_img = self.img_fusion(masked_img, img_recon)
        comp_ehr = self.ehr_fusion(masked_ehr, ehr_recon)
        
        return {
            "comp_img": comp_img,
            "comp_ehr": comp_ehr,
            "img_mask": img_mask,
            "ehr_mask": ehr_mask,
            "img_recon": img_recon,
            "ehr_recon": ehr_recon,
        }
