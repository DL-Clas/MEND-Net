from .heal_net import HEALNet
from .teacher_model import TeacherModel


def build_model(config: dict, disabled_modules: list = None) -> HEALNet:
    """Build HEAL-Net model from configuration.
    
    Args:
        config: Configuration dictionary.
        disabled_modules: List of module names to disable (for ablation).
            Options: ["HD-NAS", "HKD", "DFC"]
        
    Returns:
        Configured HEALNet model.
    """
    if disabled_modules is None:
        disabled_modules = []
    
    data_cfg = config.get("data", {})
    nas_cfg = config.get("hd_nas", {})
    hkd_cfg = config.get("hkd", {})
    dfc_cfg = config.get("dfc", {})
    
    model = HEALNet(
        num_classes=data_cfg.get("num_classes", 5),
        img_dim=128,
        ehr_dim=64,
        ehr_input_dim=data_cfg.get("num_ehr_features", 12),
        num_nodes=nas_cfg.get("num_nodes", 4),
        lambda_flops=nas_cfg.get("lambda_flops", 0.05),
        snr_threshold=dfc_cfg.get("snr_threshold", 0.3),
        gamma=hkd_cfg.get("gamma", 1.0),
        beta=hkd_cfg.get("beta", 0.5),
        temperature=hkd_cfg.get("temperature", 3.0),
        pretrained=data_cfg.get("pretrained", True),
    )
    
    if "HD-NAS" in disabled_modules:
        _disable_nas(model)
    if "HKD" in disabled_modules:
        _disable_hkd(model)
    if "DFC" in disabled_modules:
        _disable_dfc(model)
    
    return model


def build_teacher(config: dict) -> TeacherModel:
    """Build teacher model from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        TeacherModel instance.
    """
    data_cfg = config.get("data", {})
    
    return TeacherModel(
        num_classes=data_cfg.get("num_classes", 5),
        img_dim=768,
        ehr_dim=64,
        ehr_input_dim=data_cfg.get("num_ehr_features", 12),
        pretrained=data_cfg.get("pretrained", True),
    )


def _disable_nas(model: HEALNet) -> None:
    """Disable HD-NAS by freezing the search space weights."""
    for param in model.search_space.parameters():
        param.requires_grad = False
    model.search_space.eval()
    model.nas_disabled = True


def _disable_hkd(model: HEALNet) -> None:
    """Disable HKD by freezing the distillation module parameters."""
    for param in model.hkd.parameters():
        param.requires_grad = False
    model.hkd.eval()
    model.hkd_disabled = True


def _disable_dfc(model: HEALNet) -> None:
    """Disable DFC module."""
    for param in model.dfc.parameters():
        param.requires_grad = False
    model.dfc.eval()
    model.dfc_disabled = True
