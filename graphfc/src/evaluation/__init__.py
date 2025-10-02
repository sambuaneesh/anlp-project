"""
Evaluation module for GraphFC framework.
"""

from .metrics import (
    FactCheckingMetrics,
    evaluate_model_on_dataset,
    compare_models,
    print_evaluation_report
)

__all__ = [
    "FactCheckingMetrics",
    "evaluate_model_on_dataset", 
    "compare_models",
    "print_evaluation_report"
]