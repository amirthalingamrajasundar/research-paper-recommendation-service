"""Evaluation module for recommendation models."""
from .evaluator import (
    LLMEvaluator,
    calculate_metrics,
    calculate_metrics_at_k,
    evaluate_models,
    evaluate_models_multi_k,
    print_evaluation_summary,
    print_multi_k_summary,
    compare_models,
    compare_models_multi_k,
    plot_metrics_comparison
)

__all__ = [
    'LLMEvaluator',
    'calculate_metrics',
    'calculate_metrics_at_k',
    'evaluate_models',
    'evaluate_models_multi_k',
    'print_evaluation_summary',
    'print_multi_k_summary',
    'compare_models',
    'compare_models_multi_k',
    'plot_metrics_comparison'
]
