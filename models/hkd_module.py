import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Non-linear projection to unified metric space.
    
    Implements Eq.9-10 from HEAL-Net paper:
    F_tilde = sigma(W * F + b)
    
    Args:
        in_dim: Input feature dimension.
        out_dim: Output projection dimension.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossCovarianceAligner(nn.Module):
    """Cross-modal covariance matrix alignment module.
    
    Computes cross-modal covariance matrices for teacher and student,
    then aligns them via Frobenius norm.
    
    Implements Eq.11-12 from HEAL-Net paper:
    C^T = (1/(B-1)) * sum(F_img^T - mu_img) * (F_ehr^T - mu_ehr)^T
    L_Relation = (1/d_h^2) * ||C^T - C^S||_F^2
    """
    
    def __init__(self, proj_dim: int = 128):
        super().__init__()
        self.proj_dim = proj_dim
    
    def compute_cross_covariance(
        self,
        feat_img: torch.Tensor,
        feat_ehr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-modal covariance matrix.
        
        Args:
            feat_img: Image features (B, D).
            feat_ehr: EHR features (B, D).
            
        Returns:
            Covariance matrix (D, D).
        """
        feat_img_centered = feat_img - feat_img.mean(dim=0, keepdim=True)
        feat_ehr_centered = feat_ehr - feat_ehr.mean(dim=0, keepdim=True)
        
        B = feat_img.shape[0]
        cov = torch.mm(feat_img_centered.t(), feat_ehr_centered) / max(B - 1, 1)
        return cov
    
    def forward(
        self,
        teacher_img: torch.Tensor,
        teacher_ehr: torch.Tensor,
        student_img: torch.Tensor,
        student_ehr: torch.Tensor,
    ) -> tuple:
        """Compute cross-covariance matrices for teacher and student.
        
        Returns:
            Tuple of (teacher_cov, student_cov).
        """
        teacher_cov = self.compute_cross_covariance(teacher_img, teacher_ehr)
        student_cov = self.compute_cross_covariance(student_img, student_ehr)
        return teacher_cov, student_cov


class HKDModule(nn.Module):
    """Hierarchical Knowledge Distillation with cross-modal relation alignment.
    
    Combines:
    1. Cross-modal covariance alignment (L_Relation)
    2. KL divergence soft label distillation (L_KL)
    3. Cross-entropy classification loss (L_CE)
    
    Implements Eq.13-14 from HEAL-Net paper:
    L_Distill = L_CE + gamma * L_Relation + beta * L_KL
    """
    
    def __init__(
        self,
        teacher_img_dim: int = 768,
        teacher_ehr_dim: int = 64,
        student_img_dim: int = 128,
        student_ehr_dim: int = 64,
        proj_dim: int = 128,
        temperature: float = 3.0,
        gamma: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.beta = beta
        
        self.teacher_img_proj = ProjectionHead(teacher_img_dim, proj_dim)
        self.teacher_ehr_proj = ProjectionHead(teacher_ehr_dim, proj_dim)
        self.student_img_proj = ProjectionHead(student_img_dim, proj_dim)
        self.student_ehr_proj = ProjectionHead(student_ehr_dim, proj_dim)
        
        self.cov_aligner = CrossCovarianceAligner(proj_dim)
        
        self.proj_dim = proj_dim
    
    def relation_loss(
        self,
        teacher_img: torch.Tensor,
        teacher_ehr: torch.Tensor,
        student_img: torch.Tensor,
        student_ehr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Frobenius norm of cross-covariance difference.
        
        Implements Eq.11: L_Relation = (1/d_h^2) * ||C^T - C^S||_F^2
        
        Returns:
            Relation alignment loss scalar.
        """
        t_img = self.teacher_img_proj(teacher_img)
        t_ehr = self.teacher_ehr_proj(teacher_ehr)
        s_img = self.student_img_proj(student_img)
        s_ehr = self.student_ehr_proj(student_ehr)
        
        teacher_cov, student_cov = self.cov_aligner(t_img, t_ehr, s_img, s_ehr)
        
        d_h = self.proj_dim
        loss = torch.norm(teacher_cov - student_cov, p="fro") ** 2 / (d_h ** 2)
        return loss
    
    def kl_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between soft labels.
        
        Implements Eq.12: L_KL = KL(softmax(z_T/T) || softmax(z_S/T))
        
        Args:
            teacher_logits: Teacher model logits (B, C).
            student_logits: Student model logits (B, C).
            
        Returns:
            KL divergence loss.
        """
        t_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        s_log_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        loss = F.kl_div(s_log_soft, t_soft, reduction="batchmean")
        return loss * (self.temperature ** 2)
    
    def forward(
        self,
        teacher_img_feat: torch.Tensor,
        teacher_ehr_feat: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_img_feat: torch.Tensor,
        student_ehr_feat: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Compute full distillation loss.
        
        Returns:
            Dict with total loss, relation_loss, kl_loss, ce_loss.
        """
        ce_loss = F.cross_entropy(student_logits, labels)
        rel_loss = self.relation_loss(
            teacher_img_feat, teacher_ehr_feat,
            student_img_feat, student_ehr_feat,
        )
        kl = self.kl_loss(teacher_logits, student_logits)
        
        total = ce_loss + self.gamma * rel_loss + self.beta * kl
        
        return {
            "loss": total,
            "ce_loss": ce_loss,
            "relation_loss": rel_loss,
            "kl_loss": kl,
        }
