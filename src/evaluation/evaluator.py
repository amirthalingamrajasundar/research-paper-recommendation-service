"""
LLM-based evaluation for recommendation models.
"""
import os
import asyncio
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings

load_dotenv()


class ScoredPaper(BaseModel):
    """Structured output for paper relevance scoring."""
    score: int = Field(description="Relevance score (1-5)", ge=1, le=5)
    reasoning: str = Field(description="Reasoning for the score")


class LLMEvaluator:
    """Evaluator that uses LLMs to score recommendation quality."""
    
    def __init__(self):
        self.llms = {
            'gpt-4o': ChatOpenAI(
                model=os.getenv("GPT_4O"),
                temperature=0.1,
                openai_api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            ).with_structured_output(ScoredPaper),
            'claude-3.5-haiku': ChatOpenAI(
                model=os.getenv("CLAUDE_3_5_HAIKU"),
                temperature=0.1,
                openai_api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            ).with_structured_output(ScoredPaper),
        }
    
    def _build_prompt(self, query_paper: Dict, recommendation: Dict) -> str:
        """Build the evaluation prompt."""
        return textwrap.dedent(f"""
            You are an expert research paper recommendation evaluator. Evaluate the relevance of the given recommended paper to the given query paper.
            
            QUERY PAPER:
                Title: {query_paper.get('title', '')}
                Abstract: {query_paper.get('abstract', '')}
                Categories: {query_paper.get('categories', '')}
            
            RECOMMENDATION TO EVALUATE:
                Title: {recommendation.get('title', '')}
                Abstract: {recommendation.get('abstract', '')}
                Categories: {recommendation.get('categories', '')}
                
            SCORING CRITERIA:
                5 = Highly relevant (same specific topic, methods, or direct extension)
                4 = Relevant (closely related field, similar techniques)  
                3 = Moderately relevant (same broad area, some conceptual overlap)
                2 = Weakly relevant (distant connection, different subfield)
                1 = Not relevant (unrelated topic)
            
            Consider:
                - Topic alignment and research questions
                - Methodological similarity
                - Shared keywords and concepts
                - Potential for citation
                - Same problem domain
        """).strip()
    
    def get_score_for_recommendation(self, llm, query_paper: Dict, recommendation: Dict) -> ScoredPaper:
        """Get score for a single recommendation using one LLM."""
        prompt = self._build_prompt(query_paper, recommendation)
        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"LLM error: {e}")
            return ScoredPaper(score=3, reasoning="Error - defaulting to neutral score")
    
    def get_scores(self, query_paper: Dict, recommendations: List[Dict]) -> List[float]:
        """
        Get scores for all recommendations using LLM ensemble.
        
        Returns:
            List of average scores (1-5) for each recommendation
        """
        all_scores = defaultdict(dict)
        avg_scores = []
        
        for rec in recommendations:
            score_sum = 0
            for llm_name, llm in self.llms.items():
                response = self.get_score_for_recommendation(llm, query_paper, rec)
                score_sum += response.score
                all_scores[llm_name][rec.get('id', '')] = response
            
            avg_score = score_sum / len(self.llms)
            avg_scores.append(avg_score)
        
        return avg_scores


def calculate_metrics(scores: List[float], rec_categories: List[str], 
                      query_category: str, k: int = None) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        scores: List of relevance scores (1-5) for each recommendation
        rec_categories: List of categories for each recommendation
        query_category: Category of the query paper
        k: Number of recommendations to consider (default: len(scores))
    
    Returns:
        Dictionary of metric names to values
    """
    # Use actual number of scores if k not specified
    k = k or len(scores)
    scores = np.array(scores[:k])
    k = len(scores)  # Ensure k matches actual scores length
    binary = (scores >= 3).astype(int)  # Relevant if score >= 3
    
    # Basic metrics
    precision = np.mean(binary)
    recall = np.sum(binary) / min(10, k)  # Assume max 10 relevant exist
    
    metrics = {
        'precision_at_k': precision,
        'recall_at_k': recall,
        'f1_at_k': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'avg_relevance': np.mean(scores),
        'category_consistency': np.mean([1 if cat == query_category else 0 for cat in rec_categories[:k]])
    }
    
    # MRR (Mean Reciprocal Rank)
    if np.any(binary):
        metrics['mrr'] = 1.0 / (np.argmax(binary) + 1)
    else:
        metrics['mrr'] = 0.0
    
    # MAP@K (Mean Average Precision)
    relevant_positions = np.where(binary)[0]
    if len(relevant_positions) > 0:
        precisions = [(i+1) / (pos+1) for i, pos in enumerate(relevant_positions)]
        metrics['map_at_k'] = np.mean(precisions)
    else:
        metrics['map_at_k'] = 0.0
    
    # nDCG@K (Normalized Discounted Cumulative Gain)
    dcg = np.sum((2**scores - 1) / np.log2(np.arange(1, k+1) + 1))
    ideal = np.sum((2**np.sort(scores)[::-1] - 1) / np.log2(np.arange(1, k+1) + 1))
    metrics['ndcg_at_k'] = dcg / ideal if ideal > 0 else 0.0
    
    return metrics


def evaluate_models(models: Dict[str, Any], df: pd.DataFrame, 
                    n_samples: int = 20, top_k: int = 10,
                    random_seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Evaluate multiple recommendation models using LLM-based scoring.
    
    Args:
        models: Dictionary of model_name -> model object
        df: DataFrame with all papers
        n_samples: Number of random query papers to sample
        top_k: Number of recommendations per query
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary of model_name -> DataFrame of metrics per sample
    """
    np.random.seed(random_seed)
    evaluator = LLMEvaluator()
    eval_indices = np.random.choice(len(df), n_samples, replace=False)
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        model_results = []
        
        for idx in tqdm(eval_indices, desc=f"Evaluating {model_name}"):
            # Get recommendations
            recs_df = model.get_recommendations_for_paper(idx, top_k=top_k)
            
            # Prepare data
            query = df.iloc[idx].to_dict()
            recs = [row['paper'] if isinstance(row['paper'], dict) else row['paper'].to_dict() 
                    for _, row in recs_df.iterrows()]
            rec_categories = [r.get('primary_category', '') for r in recs]
            
            # Get scores from LLMs
            scores = evaluator.get_scores(query, recs)
            
            # Calculate all metrics
            metrics = calculate_metrics(
                scores, rec_categories, 
                query.get('primary_category', ''), 
                k=top_k
            )
            model_results.append(metrics)
        
        all_results[model_name] = pd.DataFrame(model_results)
    
    return all_results


def print_evaluation_summary(results: Dict[str, pd.DataFrame]):
    """Print a summary of evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for model_name, results_df in results.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        for metric in results_df.columns:
            mean = results_df[metric].mean()
            std = results_df[metric].std()
            print(f"  {metric:20s}: {mean:.3f} ± {std:.3f}")


def compare_models(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a comparison DataFrame of all models.
    
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    comparison = []
    for model_name, results_df in results.items():
        row = {'model': model_name}
        for metric in results_df.columns:
            row[f'{metric}_mean'] = results_df[metric].mean()
            row[f'{metric}_std'] = results_df[metric].std()
        comparison.append(row)
    
    return pd.DataFrame(comparison).set_index('model')


def calculate_metrics_at_k(scores: List[float], rec_categories: List[str], 
                           query_category: str, k_values: List[int] = None) -> Dict[str, Dict[int, float]]:
    """
    Calculate metrics for multiple k values.
    
    Args:
        scores: List of relevance scores (1-5) for each recommendation
        rec_categories: List of categories for each recommendation
        query_category: Category of the query paper
        k_values: List of k values to evaluate
    
    Returns:
        Dictionary of metric_name -> {k: value}
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    scores = np.array(scores)
    metrics_by_k = {
        'precision': {},
        'recall': {},
        'f1': {},
        'avg_relevance': {},
        'mrr': {},
        'map': {},
        'ndcg': {},
        'category_consistency': {}
    }
    
    for k in k_values:
        if k > len(scores):
            continue
            
        scores_k = scores[:k]
        binary = (scores_k >= 3).astype(int)
        
        precision = np.mean(binary)
        recall = np.sum(binary) / min(10, k)
        
        metrics_by_k['precision'][k] = precision
        metrics_by_k['recall'][k] = recall
        metrics_by_k['f1'][k] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics_by_k['avg_relevance'][k] = np.mean(scores_k)
        metrics_by_k['category_consistency'][k] = np.mean([1 if cat == query_category else 0 for cat in rec_categories[:k]])
        
        # MRR
        if np.any(binary):
            metrics_by_k['mrr'][k] = 1.0 / (np.argmax(binary) + 1)
        else:
            metrics_by_k['mrr'][k] = 0.0
        
        # MAP@K
        relevant_positions = np.where(binary)[0]
        if len(relevant_positions) > 0:
            precisions_at_pos = [(i+1) / (pos+1) for i, pos in enumerate(relevant_positions)]
            metrics_by_k['map'][k] = np.mean(precisions_at_pos)
        else:
            metrics_by_k['map'][k] = 0.0
        
        # nDCG@K
        dcg = np.sum((2**scores_k - 1) / np.log2(np.arange(1, k+1) + 1))
        ideal = np.sum((2**np.sort(scores_k)[::-1] - 1) / np.log2(np.arange(1, k+1) + 1))
        metrics_by_k['ndcg'][k] = dcg / ideal if ideal > 0 else 0.0
    
    return metrics_by_k


def evaluate_models_multi_k(models: Dict[str, Any], df: pd.DataFrame, 
                            n_samples: int = 20, k_values: List[int] = None,
                            random_seed: int = 42) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Evaluate models across multiple k values.
    
    Args:
        models: Dictionary of model_name -> model object
        df: DataFrame with all papers
        n_samples: Number of random query papers to sample
        k_values: List of k values to evaluate (default: [1, 3, 5, 10])
        random_seed: Random seed for reproducibility
    
    Returns:
        {model_name: {metric_name: DataFrame with k as columns, samples as rows}}
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    max_k = max(k_values)
    np.random.seed(random_seed)
    evaluator = LLMEvaluator()
    eval_indices = np.random.choice(len(df), n_samples, replace=False)
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics_samples = []  # List of metrics_by_k for each sample
        
        for idx in tqdm(eval_indices, desc=f"Evaluating {model_name}"):
            recs_df = model.get_recommendations_for_paper(idx, top_k=max_k)
            
            query = df.iloc[idx].to_dict()
            recs = [row['paper'] if isinstance(row['paper'], dict) else row['paper'].to_dict() 
                    for _, row in recs_df.iterrows()]
            rec_categories = [r.get('primary_category', '') for r in recs]
            
            scores = evaluator.get_scores(query, recs)
            metrics_by_k = calculate_metrics_at_k(
                scores, rec_categories, 
                query.get('primary_category', ''), 
                k_values=k_values
            )
            metrics_samples.append(metrics_by_k)
        
        # Aggregate: for each metric, create DataFrame with samples as rows, k as columns
        model_metrics = {}
        for metric in metrics_samples[0].keys():
            data = {k: [sample[metric].get(k, np.nan) for sample in metrics_samples] for k in k_values}
            model_metrics[metric] = pd.DataFrame(data)
        
        all_results[model_name] = model_metrics
    
    return all_results


def print_multi_k_summary(results: Dict[str, Dict[str, pd.DataFrame]], k_values: List[int] = None):
    """
    Print summary of multi-k evaluation results.
    
    Args:
        results: Output from evaluate_models_multi_k
        k_values: List of k values (inferred from results if not provided)
    """
    print("\n" + "="*80)
    print("MULTI-K EVALUATION SUMMARY")
    print("="*80)
    
    for model_name, metrics_dict in results.items():
        print(f"\n{model_name}:")
        print("-" * 60)
        
        # Get k values from first metric
        first_metric = list(metrics_dict.keys())[0]
        k_values = list(metrics_dict[first_metric].columns)
        
        # Header
        header = f"{'Metric':<25}" + "".join([f"k={k:<8}" for k in k_values])
        print(header)
        print("-" * 60)
        
        for metric_name, metric_df in metrics_dict.items():
            row = f"{metric_name:<25}"
            for k in k_values:
                mean = metric_df[k].mean()
                row += f"{mean:<8.3f}"
            print(row)


def plot_metrics_comparison(results: Dict[str, Dict[str, pd.DataFrame]], 
                            output_path: Optional[str] = None, 
                            metrics_to_plot: List[str] = None,
                            figsize: tuple = (14, 10)):
    """
    Plot comparison graphs for each metric across k values.
    
    Args:
        results: Output from evaluate_models_multi_k
        output_path: Path to save the plot (if None, displays interactively)
        metrics_to_plot: List of metrics to plot (default: precision, ndcg, map, avg_relevance)
        figsize: Figure size as (width, height)
    
    Returns:
        matplotlib Figure object
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['precision', 'ndcg', 'map', 'avg_relevance', 'mrr', 'f1']
    
    # Filter to only metrics that exist
    model_names = list(results.keys())
    available_metrics = list(results[model_names[0]].keys())
    metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
    
    if len(metrics_to_plot) == 0:
        print("No valid metrics to plot.")
        return None
    
    k_values = list(results[model_names[0]][metrics_to_plot[0]].columns)
    
    # Color scheme
    colors = {
        'tfidf': '#FF6B6B', 
        'sentence_transformer': '#4ECDC4', 
        'finetuned_st': '#45B7D1'
    }
    default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Calculate grid dimensions
    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for ax_idx, (ax, metric) in enumerate(zip(axes, metrics_to_plot)):
        for model_idx, model_name in enumerate(model_names):
            means = results[model_name][metric].mean()
            stds = results[model_name][metric].std()
            
            color = colors.get(model_name, default_colors[model_idx % len(default_colors)])
            ax.plot(k_values, means, marker='o', label=model_name, color=color, 
                   linewidth=2, markersize=8)
            ax.fill_between(k_values, means - stds, means + stds, alpha=0.2, color=color)
        
        ax.set_xlabel('k', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()}@k', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
    
    # Hide empty subplots
    for ax_idx in range(len(metrics_to_plot), len(axes)):
        axes[ax_idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()
    return fig


def compare_models_multi_k(results: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Create a comparison DataFrame for multi-k results.
    
    Args:
        results: Output from evaluate_models_multi_k
    
    Returns:
        DataFrame with hierarchical columns (metric, k) and models as rows
    """
    model_names = list(results.keys())
    first_model = results[model_names[0]]
    metrics = list(first_model.keys())
    k_values = list(first_model[metrics[0]].columns)
    
    rows = []
    for model_name in model_names:
        row = {'model': model_name}
        for metric in metrics:
            for k in k_values:
                mean = results[model_name][metric][k].mean()
                std = results[model_name][metric][k].std()
                row[f'{metric}@{k}'] = f"{mean:.3f}"
        rows.append(row)
    
    return pd.DataFrame(rows).set_index('model')


if __name__ == "__main__":
    import pickle
    from src.models.tfidf.trainer import TFIDFModel
    from src.models.sentence_transformer.base_model import SentenceTransformerModel
    
    # Load data
    df = pd.read_parquet(settings.data.dataset.paths.processed_data)
    print(f"Loaded {len(df)} papers")
    
    # Load models
    print("\nLoading models...")
    tfidf_model = TFIDFModel()
    tfidf_model.load(df)
    
    st_model = SentenceTransformerModel()
    st_model.load(df)
    
    models = {
        'tfidf': tfidf_model,
        'sentence_transformer': st_model,
    }
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate_models(models, df, n_samples=5, top_k=5)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results
    output_path = settings.model.tfidf.paths.vectorizer.parent.parent / "eval_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")
