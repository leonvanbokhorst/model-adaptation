import torch
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    bits_per_byte: float
    perplexity: float
    uncertainty: float


class MetricsComputer:
    """Compute and track evaluation metrics."""

    def __init__(self):
        self.metrics_history = []

    def compute_bits_per_byte(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Compute bits per byte metric."""
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
        )
        return (loss / np.log(2)).item()

    def compute_perplexity(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute perplexity."""
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
        )
        return torch.exp(loss).item()

    def compute_metrics(
        self, 
        outputs: Dict[str, Any], 
        labels: Optional[torch.Tensor] = None, 
        uncertainty: Optional[float] = None
    ) -> Dict[str, float]:
        """Compute training metrics."""
        metrics = {
            'loss': outputs['loss'],
            'uncertainty': uncertainty if uncertainty is not None else 0.0
        }
        
        if outputs['logits'] is not None and labels is not None:
            # Add any additional metrics computation here
            pass
            
        return metrics

    def get_metrics_summary(self) -> Dict[str, List[float]]:
        """Get summary of tracked metrics."""
        return {
            "bits_per_byte": [m.bits_per_byte for m in self.metrics_history],
            "perplexity": [m.perplexity for m in self.metrics_history],
            "uncertainty": [m.uncertainty for m in self.metrics_history],
        }


class AdaptiveStoppingMetrics:
    """Track metrics for adaptive stopping."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.uncertainty_history = []
        self.compute_history = []

    def should_stop(self, uncertainty: float, step: int) -> bool:
        """Determine if training should stop based on uncertainty."""
        if step < 5:  # Minimum number of steps
            return False
        
        # Add your stopping criterion here
        return uncertainty < self.alpha

    def get_stopping_summary(self) -> Dict[str, list]:
        """Get summary of stopping metrics."""
        return {
            "uncertainty": self.uncertainty_history,
            "compute": self.compute_history,
        }
