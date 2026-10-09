import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.metrics import compute_all_metrics, compute_flops


class Evaluator:
    """Evaluation engine for MEND-Net models.
    
    Supports standard evaluation with per-sample recording,
    confidence scoring, and hardware efficiency profiling.
    
    Args:
        model: Trained model to evaluate.
        test_loader: Test data loader.
        config: Configuration dictionary.
    """
    
    def __init__(self, model: nn.Module, test_loader, config: dict):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self.num_classes = config.get("data", {}).get("num_classes", 5)
        self.class_names = config.get("data", {}).get("class_names", [])
    
    @torch.no_grad()
    def evaluate(self, missing_rate: float = 0.0) -> dict:
        """Run full evaluation on test set.
        
        Args:
            missing_rate: Probability of masking the EHR modality per
                sample (0.0 = complete input, 0.5 = half the samples
                missing EHR). Used to measure DFC robustness.
            
        Returns:
            Dict with accuracy, f1, auc, and per-sample predictions.
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
            images = batch["image"].to(self.device)
            ehr = batch["ehr"].to(self.device)
            labels = batch["label"]
            ehr_mask = batch["ehr_mask"].to(self.device)
            img_mask = batch["img_mask"].to(self.device)
            
            if missing_rate > 0:
                drop = torch.rand(ehr.shape[0], device=ehr.device) < missing_rate
                ehr_mask = ehr_mask.clone()
                ehr_mask[drop] = 0.0
            
            logits = self.model(images, ehr, ehr_mask, img_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu()
            
            all_preds.append(preds.numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
        
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        
        metrics = compute_all_metrics(y_true, y_pred, y_prob, self.num_classes)
        metrics["predictions"] = y_pred.tolist()
        metrics["labels"] = y_true.tolist()
        metrics["probabilities"] = y_prob.tolist()
        
        return metrics
    
    @torch.no_grad()
    def evaluate_with_confidence(self) -> dict:
        """Evaluate with confidence scores for each prediction.
        
        Returns:
            Dict with metrics and per-sample confidence values.
        """
        self.model.eval()
        
        results = self.evaluate()
        
        probs = np.array(results["probabilities"])
        preds = np.array(results["predictions"])
        confidences = probs[np.arange(len(preds)), preds]
        
        results["confidences"] = confidences.tolist()
        results["mean_confidence"] = float(np.mean(confidences))
        results["std_confidence"] = float(np.std(confidences))
        
        return results
    
    def compute_efficiency(self, input_shapes: dict = None) -> dict:
        """Compute model efficiency metrics.
        
        Args:
            input_shapes: Dict mapping input names to shapes.
            
        Returns:
            Dict with params, flops.
        """
        if input_shapes is None:
            batch_size = self.config.get("training", {}).get("batch_size", 1)
            input_shapes = {
                "images": (1, 3, 224, 224),
                "ehr": (1, 12),
                "ehr_mask": (1,),
                "img_mask": (1,),
            }
        
        num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        flops = compute_flops(self.model, input_shapes, self.device)
        
        return {
            "params_m": round(num_params, 2),
            "flops_g": round(flops, 2),
        }
    
    def save_results(self, results: dict, output_dir: str = "./results") -> None:
        """Save evaluation results to JSON.
        
        Args:
            results: Evaluation results dictionary.
            output_dir: Output directory path.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        save_results = {k: v for k, v in results.items()
                        if k not in ("predictions", "labels", "probabilities", "confidences")}
        
        output_path = os.path.join(output_dir, "metrics.json")
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2)
        
        print(f"Results saved to {output_path}")
