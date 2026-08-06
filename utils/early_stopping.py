import numpy as np
import torch


class EarlyStopping:
    """Early stopping mechanism to prevent overfitting.
    
    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs.
    
    Args:
        patience: Number of epochs to wait for improvement.
        path: File path to save the best model checkpoint.
        min_delta: Minimum improvement threshold.
    """
    
    def __init__(self, patience: int = 20, path: str = "best_model.pth", min_delta: float = 0.0):
        self.patience = patience
        self.path = path
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        """Check validation loss and save model if improved.
        
        Args:
            val_loss: Current epoch validation loss.
            model: Model to potentially save.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Save model checkpoint when validation loss improves.
        
        Args:
            val_loss: Current validation loss.
            model: Model to save.
        """
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
