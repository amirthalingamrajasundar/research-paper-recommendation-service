# scripts/evaluate_vs_ada.py
"""
Evaluate models against ada-002 as ground truth (no LLM calls).

Measures how well each model's recommendations match ada-002's recommendations.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_top_k_indices(similarities: np.ndarray, idx: int, k: int) -> np.ndarray:
    """Get top-k most similar indices (excluding self)."""
    sims = similarities[idx].copy()
    sims[idx] = -np.inf
    return np.argsort(sims)[::-1][:k]


def calculate_metrics_at_k(ada_top_k: np.ndarray, model_ranking: np.ndarray, k: int) -> dict:
    """Calculate metrics at a specific k."""
    ada_set = set(ada_top_k[:k])
    model_set = set(model_ranking[:k])
    
    overlap = len(ada_set & model_set)
    recall = overlap / k
    
    # NDCG
    relevances = [1.0 if idx in ada_set else 0.0 for idx in model_ranking[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ada_set))))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    # MRR - position of first relevant item
    mrr = 0.0
    for i, idx in enumerate(model_ranking[:k]):
        if idx in ada_set:
            mrr = 1.0 / (i + 1)
            break
    
    # MAP
    precisions = []
    relevant_count = 0
    for i, idx in enumerate(model_ranking[:k]):
        if idx in ada_set:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    map_score = np.mean(precisions) if precisions else 0.0
    
    return {
        'recall': recall,
        'ndcg': ndcg,
        'mrr': mrr,
        'map': map_score,
    }


def evaluate_against_ada(
    n_samples: int = 100,
    k_values: list = None,
    seed: int = 42,
):
    """Evaluate all models against ada-002 as ground truth."""
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    np.random.seed(seed)
    
    # Load holdout data
    holdout_path = settings.data.dataset.paths.holdout_data
    logger.info(f"Loading holdout data from {holdout_path}")
    holdout_df = pd.read_parquet(holdout_path)
    n_papers = len(holdout_df)
    logger.info(f"Loaded {n_papers} papers")
    
    # Load embeddings
    models = {}
    
    # Ada-002 (ground truth)
    ada_path = PROJECT_ROOT / "models/ada_embeddings/holdout_embeddings.npy"
    if not ada_path.exists():
        raise FileNotFoundError(f"Ada embeddings not found: {ada_path}")
    ada_embeddings = np.load(ada_path)
    logger.info(f"Loaded ada-002 embeddings: {ada_embeddings.shape}")
    
    # TF-IDF
    tfidf_path = PROJECT_ROOT / "models/tfidf/holdout_embeddings.npy"
    if tfidf_path.exists():
        models['tfidf'] = np.load(tfidf_path)
        logger.info(f"Loaded TF-IDF embeddings")
    
    # Base ST
    base_path = PROJECT_ROOT / "models/sentence_transformer/holdout_embeddings.npy"
    if base_path.exists():
        models['base_st'] = np.load(base_path)
        logger.info(f"Loaded base ST embeddings")
    
    # Fine-tuned ST
    finetuned_path = PROJECT_ROOT / "models/finetuned_st/holdout_embeddings.npy"
    if finetuned_path.exists():
        models['finetuned_st'] = np.load(finetuned_path)
        logger.info(f"Loaded fine-tuned ST embeddings")
    
    if not models:
        raise ValueError("No model embeddings found")
    
    logger.info(f"Evaluating models: {list(models.keys())}")
    
    # Precompute similarity matrices
    logger.info("Computing similarity matrices...")
    ada_sims = cosine_similarity(ada_embeddings)
    model_sims = {name: cosine_similarity(emb) for name, emb in models.items()}
    
    # Sample query indices
    eval_indices = np.random.choice(n_papers, min(n_samples, n_papers), replace=False)
    max_k = max(k_values)
    logger.info(f"Evaluating {len(eval_indices)} queries at k={k_values}")
    
    # Evaluate each model at each k
    # Structure: {model: {k: {metric: [values]}}}
    all_results = {model: {k: {'recall': [], 'ndcg': [], 'mrr': [], 'map': []} 
                          for k in k_values} 
                  for model in models}
    
    for idx in tqdm(eval_indices, desc="Evaluating"):
        # Get ada's ranking (ground truth)
        ada_ranking = get_top_k_indices(ada_sims, idx, max_k)
        
        for model_name in models:
            # Get model's ranking
            model_ranking = get_top_k_indices(model_sims[model_name], idx, max_k)
            
            # Calculate metrics at each k
            for k in k_values:
                metrics = calculate_metrics_at_k(ada_ranking, model_ranking, k)
                for metric, value in metrics.items():
                    all_results[model_name][k][metric].append(value)
    
    # Compute means and stds
    summary = {model: {k: {} for k in k_values} for model in models}
    for model in models:
        for k in k_values:
            for metric in ['recall', 'ndcg', 'mrr', 'map']:
                values = all_results[model][k][metric]
                summary[model][k][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                }
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (vs Ada-002 as Ground Truth)")
    print("=" * 70)
    
    for model in models:
        print(f"\n{model}:")
        print(f"  {'k':<6} {'Recall':<15} {'NDCG':<15} {'MRR':<15} {'MAP':<15}")
        print(f"  {'-'*6} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
        for k in k_values:
            s = summary[model][k]
            print(f"  {k:<6} "
                  f"{s['recall']['mean']:.3f}±{s['recall']['std']:.3f}    "
                  f"{s['ndcg']['mean']:.3f}±{s['ndcg']['std']:.3f}    "
                  f"{s['mrr']['mean']:.3f}±{s['mrr']['std']:.3f}    "
                  f"{s['map']['mean']:.3f}±{s['map']['std']:.3f}")
    
    # Save and plot
    output_dir = PROJECT_ROOT / "results/ada_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_metrics_vs_k(summary, k_values, output_dir)
    
    # Save detailed results
    for model in models:
        rows = []
        for k in k_values:
            row = {'k': k}
            for metric in ['recall', 'ndcg', 'mrr', 'map']:
                row[f'{metric}_mean'] = summary[model][k][metric]['mean']
                row[f'{metric}_std'] = summary[model][k][metric]['std']
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / f"{model}_results.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")
    
    return summary


def plot_metrics_vs_k(summary: dict, k_values: list, output_dir: Path):
    """Plot metrics vs k for all models."""
    import matplotlib.pyplot as plt
    
    metrics = ['recall', 'ndcg', 'mrr', 'map']
    metric_titles = ['Recall@k', 'NDCG@k', 'MRR@k', 'MAP@k']
    
    colors = {'tfidf': '#FF6B6B', 'base_st': '#4ECDC4', 'finetuned_st': '#45B7D1'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, metric, title in zip(axes, metrics, metric_titles):
        for model in summary:
            means = [summary[model][k][metric]['mean'] for k in k_values]
            stds = [summary[model][k][metric]['std'] for k in k_values]
            
            color = colors.get(model, 'gray')
            ax.plot(k_values, means, 'o-', label=model, color=color, linewidth=2, markersize=8)
            ax.fill_between(k_values, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           color=color, alpha=0.2)
        
        ax.set_xlabel('k', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1.05)
    
    plt.suptitle('Model Comparison vs Ada-002 (Ground Truth)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_vs_k.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot saved to {output_dir / 'metrics_vs_k.png'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate models vs ada-002")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of query samples")
    parser.add_argument("--k-values", type=int, nargs='+', default=[1, 3, 5, 10], help="Values of k to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    evaluate_against_ada(
        n_samples=args.n_samples,
        k_values=args.k_values,
        seed=args.seed,
    )
