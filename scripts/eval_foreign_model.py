# scripts/eval_foreign_model.py
"""
Evaluate a foreign fine-tuned model against our existing models.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.evaluation import LLMEvaluator, calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for embedding-based recommendation."""
    
    def __init__(self, name: str, embeddings: np.ndarray, df: pd.DataFrame):
        self.name = name
        self.embeddings = embeddings
        self.df = df
    
    def get_recommendations(self, idx: int, top_k: int = 10) -> pd.DataFrame:
        query_embedding = self.embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        similarities[idx] = -1  # Exclude self
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return pd.DataFrame([
            {'paper': self.df.iloc[i].to_dict(), 'similarity': similarities[i]}
            for i in top_indices
        ])


def main(
    foreign_model_path: str = "foreign_finetuned_st",
    n_samples: int = 10,
    top_k: int = 3,
    seed: int = 42,
):
    np.random.seed(seed)
    
    # Load holdout data
    holdout_path = settings.data.dataset.paths.holdout_data
    logger.info(f"Loading holdout data from {holdout_path}")
    holdout_df = pd.read_parquet(holdout_path)
    logger.info(f"Loaded {len(holdout_df)} papers")
    
    # Generate embeddings for foreign model
    logger.info(f"Loading foreign model from {foreign_model_path}")
    foreign_model = SentenceTransformer(foreign_model_path)
    
    logger.info("Generating embeddings for foreign model...")
    foreign_embeddings = foreign_model.encode(
        holdout_df['text'].tolist(), 
        show_progress_bar=True
    )
    
    # Load models to compare
    models = {}
    
    # Foreign model
    models['foreign_st'] = EmbeddingModel('foreign_st', foreign_embeddings, holdout_df)
    
    # Base ST (for comparison)
    base_path = PROJECT_ROOT / "models/sentence_transformer/holdout_embeddings.npy"
    if base_path.exists():
        models['base_st'] = EmbeddingModel('base_st', np.load(base_path), holdout_df)
    
    # Our finetuned (for comparison)
    finetuned_path = PROJECT_ROOT / "models/finetuned_st/holdout_embeddings.npy"
    if finetuned_path.exists():
        models['our_finetuned'] = EmbeddingModel('our_finetuned', np.load(finetuned_path), holdout_df)
    
    logger.info(f"Evaluating models: {list(models.keys())}")
    
    # Sample queries
    eval_indices = np.random.choice(len(holdout_df), min(n_samples, len(holdout_df)), replace=False)
    
    # Evaluate
    evaluator = LLMEvaluator()
    all_results = {}
    
    for model_name, model in models.items():
        logger.info(f"\n{'='*50}\nEvaluating {model_name}\n{'='*50}")
        
        model_results = []
        for idx in tqdm(eval_indices, desc=f"Evaluating {model_name}"):
            recs_df = model.get_recommendations(idx, top_k=top_k)
            
            query = holdout_df.iloc[idx].to_dict()
            recs = [row['paper'] for _, row in recs_df.iterrows()]
            rec_categories = [r.get('primary_category', '') for r in recs]
            
            try:
                scores = evaluator.get_scores(query, recs)
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}")
                scores = [3.0] * len(recs)
            
            metrics = calculate_metrics(
                scores=scores,
                query_category=query.get('primary_category', ''),
                rec_categories=rec_categories,
            )
            metrics['query_idx'] = idx
            model_results.append(metrics)
        
        results_df = pd.DataFrame(model_results)
        all_results[model_name] = results_df
        
        # Print summary
        logger.info(f"{model_name}: precision={results_df['precision_at_k'].mean():.3f}, "
                   f"avg_relevance={results_df['avg_relevance'].mean():.3f}")
    
    # Final comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for name, df in all_results.items():
        print(f"\n{name}:")
        print(f"  Precision@{top_k}: {df['precision_at_k'].mean():.3f} ± {df['precision_at_k'].std():.3f}")
        print(f"  Avg Relevance:    {df['avg_relevance'].mean():.3f} ± {df['avg_relevance'].std():.3f}")
        print(f"  NDCG@{top_k}:       {df['ndcg_at_k'].mean():.3f} ± {df['ndcg_at_k'].std():.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="foreign_finetuned_st", help="Path to foreign model")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(
        foreign_model_path=args.model_path,
        n_samples=args.n_samples,
        top_k=args.top_k,
        seed=args.seed,
    )