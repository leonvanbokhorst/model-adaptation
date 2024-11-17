import numpy as np
from typing import Dict, List, Any, Optional
import logging
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MetricsComputer:
    def __init__(self):
        """Initialize metrics tracking."""
        self.metrics_history = {
            'loss': [],
            'perplexity': [],
            'bits_per_byte': [],
            'uncertainty': []
        }
        self.step_count = 0
    
    def compute_metrics(self, outputs: Dict[str, Any], uncertainty: Optional[float] = None) -> Dict[str, float]:
        """Compute metrics from model outputs."""
        try:
            if outputs is None:
                return None
            
            metrics = {}
            
            # Extract basic metrics
            if 'loss' in outputs:
                metrics['loss'] = outputs['loss']
            if 'perplexity' in outputs:
                metrics['perplexity'] = outputs['perplexity']
            if 'bits_per_byte' in outputs:
                metrics['bits_per_byte'] = outputs['bits_per_byte']
            
            # Add uncertainty if provided
            if uncertainty is not None:
                metrics['uncertainty'] = uncertainty
            elif 'uncertainty' in outputs:
                metrics['uncertainty'] = outputs['uncertainty']
            
            # Update history
            self.update(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return None
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update metrics history with new values."""
        try:
            if metrics is None:
                return
                
            for key in self.metrics_history:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, (int, float, np.number)):
                        self.metrics_history[key].append(float(value))
                    else:
                        logger.warning(f"Skipping non-numeric metric value for {key}: {value}")
            
            self.step_count += 1
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get the most recent metrics."""
        return {
            key: values[-1] if values else float('inf')
            for key, values in self.metrics_history.items()
        }
    
    def get_metrics_summary(self) -> Dict[str, List[float]]:
        """Get the complete metrics history."""
        return self.metrics_history
    
    def get_average_metrics(self, window_size: int = 10) -> Dict[str, float]:
        """Get average metrics over recent window."""
        averages = {}
        for key, values in self.metrics_history.items():
            if values:
                recent_values = values[-window_size:]
                averages[key] = sum(recent_values) / len(recent_values)
            else:
                averages[key] = float('inf')
        return averages

class AdaptiveStoppingMetrics:
    def __init__(self, alpha: float = 0.1, window_size: int = 5):
        self.alpha = alpha
        self.window_size = window_size
        self.stopping_points = []
        self.uncertainty_history = []
    
    def should_stop(self, uncertainty: float, step: int, loss_history: List[float]) -> bool:
        """Enhanced stopping criterion using both uncertainty and loss trends."""
        try:
            self.uncertainty_history.append(uncertainty)
            
            if step < self.window_size:
                return False
            
            # Get recent metrics
            recent_uncertainties = self.uncertainty_history[-self.window_size:]
            recent_losses = loss_history[-self.window_size:]
            
            # Compute trends
            uncertainty_trend = np.polyfit(range(len(recent_uncertainties)), recent_uncertainties, 1)[0]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            # Stop if both metrics are stable or improving
            should_stop = (
                uncertainty_trend <= 0 and  # Uncertainty is decreasing
                loss_trend <= 0 and        # Loss is decreasing
                np.mean(recent_uncertainties) < self.alpha
            )
            
            if should_stop:
                self.stopping_points.append(step)
                logger.info(f"Stopping at step {step} with uncertainty {np.mean(recent_uncertainties):.4f}")
            
            return should_stop
            
        except Exception as e:
            logger.error(f"Error in stopping criterion: {e}")
            return False
    
    def get_stopping_summary(self) -> Dict[str, List[int]]:
        """Get summary of stopping points and uncertainties."""
        return {
            'stopping_points': self.stopping_points,
            'uncertainty_history': self.uncertainty_history
        } 

def compute_validation_metrics(self, validation_examples: List[str]) -> Dict[str, float]:
    self.model.eval()
    total_loss = 0.0
    total_perplexity = 0.0
    
    with torch.no_grad():
        for example in validation_examples:
            inputs = self.tokenizer(
                example,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            total_perplexity += torch.exp(outputs.loss).item()
    
    avg_metrics = {
        'val_loss': total_loss / len(validation_examples),
        'val_perplexity': total_perplexity / len(validation_examples)
    }
    
    self.model.train()
    return avg_metrics 