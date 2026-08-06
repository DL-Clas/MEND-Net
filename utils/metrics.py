import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
    Returns:
        Accuracy score.
    """
    return float(accuracy_score(y_true, y_pred))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    """Compute F1 score.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy ('macro', 'micro', 'weighted').
        
    Returns:
        F1 score.
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 5) -> float:
    """Compute macro-averaged AUC.
    
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities (N, C).
        num_classes: Number of classes.
        
    Returns:
        Macro-averaged AUC score.
    """
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        return 0.0


def compute_flops(model: torch.nn.Module, input_shapes: dict, device: torch.device) -> float:
    """Compute FLOPs using fvcore.
    
    Args:
        model: PyTorch model.
        input_shapes: Dict mapping input names to tensor shapes.
        device: Computation device.
        
    Returns:
        Total FLOPs in Giga (1e9).
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.randn(*shape).to(device)
        
        if len(dummy_inputs) == 1:
            flop_counter = FlopCountAnalysis(model, list(dummy_inputs.values())[0])
        else:
            flop_counter = FlopCountAnalysis(model, tuple(dummy_inputs.values()))
        
        total_flops = flop_counter.total()
        return float(total_flops / 1e9)
    except Exception:
        return 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 5,
) -> dict:
    """Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (N, C).
        num_classes: Number of classes.
        
    Returns:
        Dictionary with accuracy, f1, auc.
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
        "auc": compute_auc(y_true, y_prob, num_classes),
    }
