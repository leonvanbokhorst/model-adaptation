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
