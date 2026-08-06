import json
import numpy as np
from datetime import datetime, timezone


class ExperimentRecorder:
    """Records experiment metrics and outputs experiment_summary.json.
    
    Compatible with paper-innovation Phase 7 analysis format.
    
    Args:
        innovation_names: List of innovation module names.
        sota_baselines: Dict of SOTA baseline metric values.
        baseline_metrics: Dict of baseline model metrics.
        iteration_round: Current iteration round.
        dataset_name: Name of the dataset.
    """
    
    def __init__(
        self,
        innovation_names: list,
        sota_baselines: dict,
        baseline_metrics: dict,
        iteration_round: int = 1,
        dataset_name: str = "",
    ):
        self.innovation_names = innovation_names
        self.sota_baselines = sota_baselines
        self.baseline_metrics = baseline_metrics
        self.iteration_round = iteration_round
        self.dataset_name = dataset_name
        self.metrics = {}
        self.innovation_results = []
        self.failure_cases = []
    
    def record_metric(self, name: str, value) -> None:
        """Record a scalar metric or statistics dictionary.
        
        Args:
            name: Metric name (e.g., 'best_accuracy').
            value: Scalar value or dict with mean/std/n.
        """
        if isinstance(value, dict):
            self.metrics[name] = value
        else:
            self.metrics[name] = float(value)
    
    def record_ablation_batch(
        self,
        innovation_name: str,
        with_scores: list,
        without_scores: list,
        metric_name: str = "accuracy",
        claimed_improvement: str = "",
    ) -> None:
        """Record ablation results with effect size computation.
        
        Args:
            innovation_name: Name of the innovation module.
            with_scores: Scores with the module enabled.
            without_scores: Scores with the module disabled.
            metric_name: Primary metric name.
            claimed_improvement: Expected improvement description.
        """
        with_arr = np.array(with_scores, dtype=np.float64)
        without_arr = np.array(without_scores, dtype=np.float64)
        
        with_mean = float(np.mean(with_arr))
        without_mean = float(np.mean(without_arr))
        with_std = float(np.std(with_arr, ddof=1)) if len(with_arr) > 1 else 0.0
        without_std = float(np.std(without_arr, ddof=1)) if len(without_arr) > 1 else 0.0
        
        if len(with_arr) >= 3 and with_std > 0 and without_std > 0:
            pooled_std = np.sqrt((with_std**2 + without_std**2) / 2)
            cohens_d = float((with_mean - without_mean) / (pooled_std + 1e-8))
            interpretation = self._interpret_effect_size(cohens_d)
            effectiveness = self._compute_effectiveness(cohens_d)
        else:
            cohens_d = None
            interpretation = "insufficient_data"
            effectiveness = "insufficient_data"
        
        self.innovation_results.append({
            "name": innovation_name,
            f"{metric_name}_with": with_mean,
            f"{metric_name}_without": without_mean,
            f"{metric_name}_with_std": with_std,
            f"{metric_name}_without_std": without_std,
            "n": len(with_scores),
            "cohens_d": cohens_d,
            "effectiveness": effectiveness,
            "effect_size": {
                "interpretation": interpretation,
            },
            "claimed_improvement": claimed_improvement,
        })
    
    def _compute_effectiveness(self, cohens_d: float) -> str:
        if abs(cohens_d) >= 0.8:
            return "valid"
        elif abs(cohens_d) >= 0.3:
            return "partial"
        return "invalid"
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        d = abs(cohens_d)
        if d >= 0.8:
            return "large"
        elif d >= 0.5:
            return "medium"
        elif d >= 0.2:
            return "small"
        return "negligible"
    
    def record_failure(self, scenario: str, effect: str, analysis: str = "") -> None:
        """Record a failure case.
        
        Args:
            scenario: Description of the failure scenario.
            effect: Observed effect.
            analysis: Root cause analysis.
        """
        self.failure_cases.append({
            "scenario": scenario,
            "effect": effect,
            "analysis": analysis,
        })
    
    def _compute_convergence(self) -> dict:
        """Compute convergence metrics against SOTA baselines."""
        exceeds = True
        gap_to_sota = 0.0
        
        for key, sota_val in self.sota_baselines.items():
            best_key = f"best_{key}"
            if best_key in self.metrics:
                our_val = self.metrics[best_key]
                if isinstance(our_val, dict):
                    our_val = our_val.get("mean", 0.0)
                if key in ("hd95", "hd", "mae", "mse", "rmse"):
                    if our_val > sota_val:
                        exceeds = False
                else:
                    if our_val < sota_val:
                        exceeds = False
                gap_to_sota = abs(our_val - sota_val)
        
        return {
            "exceeds_sota": exceeds,
            "gap_to_sota": float(gap_to_sota),
            "performance_trend": "unknown",
        }
    
    def save(self, output_path: str = "experiment_summary.json") -> None:
        """Save experiment summary as JSON.
        
        Args:
            output_path: Output file path.
        """
        summary = {
            "overall_metrics": {
                **self.metrics,
                "sota_baselines": self.sota_baselines,
                "dataset_name": self.dataset_name,
            },
            "innovation_results": self.innovation_results,
            "convergence": self._compute_convergence(),
            "failure_cases": self.failure_cases,
            "iteration_round": self.iteration_round,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
