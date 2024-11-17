import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import numpy as np


class SIFTVisualizer:
    """Visualization utilities for SIFT experiments."""

    def __init__(self):
        plt.style.use('default')

    def plot_metrics_over_time(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Progress",
        save_path: Optional[str] = None,
    ):
        """Plot multiple metrics over time."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title)
        
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            if values:  # Only plot if we have values
                ax.plot(values, label=metric_name)
                ax.set_xlabel("Iterations")
                ax.set_ylabel(metric_name)
                ax.set_title(metric_name)
                ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_uncertainty_vs_performance(
        self,
        uncertainty: List[float],
        performance: List[float],
        save_path: Optional[str] = None,
    ):
        """Plot uncertainty vs performance metrics."""
        if uncertainty and performance:  # Only plot if we have values
            plt.figure(figsize=(8, 6))
            plt.scatter(uncertainty, performance, alpha=0.5)
            plt.xlabel("Uncertainty")
            plt.ylabel("Bits per Byte")
            plt.title("Uncertainty vs Performance")
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            plt.close()

    def plot_adaptive_stopping(
        self,
        metrics: Dict[str, list],
        alpha: float,
        title: str = "Adaptive Stopping Analysis",
        save_path: Optional[str] = None,
    ):
        """Plot adaptive stopping analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot uncertainty threshold
        compute = np.array(metrics["compute"])
        threshold = (alpha * compute) ** -1

        ax1.plot(compute, metrics["uncertainty"], label="Uncertainty")
        ax1.plot(compute, threshold, "--", label="Threshold")
        ax1.set_xlabel("Compute")
        ax1.set_ylabel("Uncertainty")
        ax1.legend()

        # Plot stopping distribution
        stops = [
            i
            for i, (u, t) in enumerate(zip(metrics["uncertainty"], threshold))
            if u > t
        ]
        ax2.hist(stops, bins=20)
        ax2.set_xlabel("Stopping Iteration")
        ax2.set_ylabel("Count")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()
