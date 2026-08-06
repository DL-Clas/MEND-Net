import torch
import torch.nn as nn
import torch.nn.functional as F


class HEALNetLoss(nn.Module):
    """Joint loss function for HEAL-Net.
    
    Implements Eq.22 from the paper:
    L_Total = L_Distill + lambda * log(FLOPs(alpha)) + rho * L_Recon
    
    During different training stages:
    - Architecture search: L_CE + lambda * log(FLOPs(alpha))
    - Distillation: L_Distill = L_CE + gamma * L_Relation + beta * L_KL
    - Compensation: L_Distill + rho * L_Recon
    
    The relation/KL components are computed by the student's HKDModule
    (which owns the projection heads and cross-covariance aligner), while
    this module provides the cross-entropy, differentiable FLOPs penalty
    and reconstruction loss (Eq.19).
    
    Args:
        gamma: Relation alignment loss weight.
        beta: KL divergence loss weight.
        lambda_flops: FLOPs penalty weight.
        rho: Reconstruction loss weight.
        temperature: Distillation temperature.
        target_flops: Optional hardware FLOPs budget used to normalize the
            penalty term (MnasNet-style relative constraint). When None the
            absolute penalty lambda * log(FLOPs(alpha)) of the paper is used.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        beta: float = 0.5,
        lambda_flops: float = 0.05,
        rho: float = 0.1,
        temperature: float = 3.0,
        target_flops: float = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.lambda_flops = lambda_flops
        self.rho = rho
        self.temperature = temperature
        self.target_flops = float(target_flops) if target_flops is not None else None
    
    def ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard cross-entropy loss."""
        return F.cross_entropy(logits, labels)
    
    def nas_loss(
        self,
        ce_loss: torch.Tensor,
        current_flops: torch.Tensor,
    ) -> torch.Tensor:
        """Hardware-aware NAS loss (Eq.6).
        
        L_NAS = L_CE + lambda * log(FLOPs(alpha))
        The FLOPs term is differentiable with respect to the architecture
        parameters, providing the hardware penalty gradient to alpha.
        When a target_flops budget is configured the term is normalized
        to log(FLOPs(alpha) / FLOPs_target).
        
        Args:
            ce_loss: Cross-entropy loss tensor.
            current_flops: Differentiable expected FLOPs tensor.
            
        Returns:
            Combined loss with hardware penalty.
        """
        flops = torch.clamp(current_flops, min=1.0)
        if self.target_flops:
            flops = flops / self.target_flops
        return ce_loss + self.lambda_flops * torch.log(flops)
    
    def reconstruction_loss(
        self,
        img_feat: torch.Tensor,
        ehr_feat: torch.Tensor,
        img_recon: torch.Tensor,
        ehr_recon: torch.Tensor,
        img_mask: torch.Tensor,
        ehr_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruction loss for DFC (Eq.19).
        
        L_Recon = sum_m (1 - M_m) / (B * d_S) * ||F_bar_m - F_m||_F^2
        
        Only the missing-modality reconstructions are penalized against the
        complete hidden features.
        """
        B, d_img = img_feat.shape
        _, d_ehr = ehr_feat.shape
        
        missing_img = (1 - img_mask).unsqueeze(-1)
        img_loss = (missing_img * (img_recon - img_feat) ** 2).sum() / (B * d_img + 1e-8)
        
        missing_ehr = (1 - ehr_mask).unsqueeze(-1)
        ehr_loss = (missing_ehr * (ehr_recon - ehr_feat) ** 2).sum() / (B * d_ehr + 1e-8)
        
        return img_loss + ehr_loss
    
    def forward(
        self,
        model,
        teacher,
        batch: dict,
        stage: str,
    ) -> dict:
        """Compute the stage-specific joint loss.
        
        Args:
            model: HEALNet student model.
            teacher: Optional teacher model (required for distillation).
            batch: Batch dict with 'image', 'ehr', 'label',
                'ehr_mask' and 'img_mask'.
            stage: Training stage ('search', 'distill', 'compensate').
            
        Returns:
            Dict with 'loss' and component losses.
        """
        images = batch["image"]
        ehr = batch["ehr"]
        labels = batch["label"]
        ehr_mask = batch["ehr_mask"]
        img_mask = batch["img_mask"]
        
        student_out = model.forward_with_features(images, ehr, ehr_mask, img_mask)
        logits = student_out["logits"]
        
        ce = self.ce_loss(logits, labels)
        result = {"loss": ce, "ce_loss": ce}
        
        if stage == "search":
            current_flops = model.search_space.expected_flops()
            result["loss"] = self.nas_loss(ce, current_flops)
            result["nas_flops"] = current_flops.detach()
            return result
        
        if teacher is not None and not getattr(model, "hkd_disabled", False):
            with torch.no_grad():
                teacher_out = teacher.forward_with_features(images, ehr)
            rel = model.hkd.relation_loss(
                teacher_out["img_feat"], teacher_out["ehr_feat"],
                student_out["raw_img_feat"], student_out["raw_ehr_feat"],
            )
            kl = model.hkd.kl_loss(teacher_out["logits"], logits)
            result["loss"] = ce + model.hkd.gamma * rel + model.hkd.beta * kl
            result["relation_loss"] = rel
            result["kl_loss"] = kl
        
        if stage == "compensate" and not getattr(model, "dfc_disabled", False):
            dfc_out = student_out["dfc_out"]
            recon = self.reconstruction_loss(
                student_out["raw_img_feat"], student_out["raw_ehr_feat"],
                dfc_out["img_recon"], dfc_out["ehr_recon"],
                dfc_out["img_mask"], dfc_out["ehr_mask"],
            )
            result["loss"] = result["loss"] + self.rho * recon
            result["recon_loss"] = recon
        
        return result
