"""
Mine hard training pairs by finding disagreements between base ST and ada-002.

Targeted approach:
1. For each paper, find ada's top-k most similar papers
2. Check if base ST disagrees (low similarity for ada's top matches)
3. These are hard positives - cases where base ST misses what ada finds

This is much more efficient than random sampling.
"""
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_or_generate_base_embeddings(df: pd.DataFrame, output_path: str) -> np.ndarray:
    """Load existing base ST embeddings or generate new ones."""
    if os.path.exists(output_path):
        logger.info(f"Loading existing base ST embeddings from {output_path}")
        return np.load(output_path)
    
    logger.info("Generating base ST embeddings...")
    model_name = settings.model.sentence_transformer.base_model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    logger.info(f"Saved base ST embeddings to {output_path}")
    
    return embeddings


def mine_hard_pairs(
    n_pairs: int = None,
    top_k: int = 20,              # Check ada's top-k matches per paper
    pos_percentile: float = 75,   # Ada sim >= this percentile is positive
    neg_percentile: float = 25,   # Ada sim <= this percentile is negative  
    gap_threshold: float = 0.2,   # Minimum gap for hard positive
    include_negatives: bool = True,
    random_seed: int = 42,
):
    """
    Mine training pairs using targeted approach with balanced positives/negatives.
    
    Uses PERCENTILE-based thresholds since ada similarities can be dense
    (e.g., all pairs might have ada_sim > 0.6 in domain-specific corpora).
    
    Args:
        n_pairs: Target number of pairs to generate
        top_k: Number of ada's top/bottom matches to check per paper
        pos_percentile: Ada sim >= this percentile is positive (default 75th)
        neg_percentile: Ada sim <= this percentile is negative (default 25th)
        gap_threshold: Minimum (ada_sim - base_sim) gap for hard positive
        include_negatives: Whether to include negative pairs (IMPORTANT for training)
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with positive AND negative pairs for balanced training
    """
    # Load defaults from config
    hp_config = settings.data.hard_pairs
    n_pairs = n_pairs or hp_config.n_pairs
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load data from config path
    train_path = settings.data.dataset.paths.train_data
    logger.info(f"Loading training data from {train_path}...")
    df = pd.read_parquet(train_path)
    n_papers = len(df)
    logger.info(f"Loaded {n_papers} papers")
    
    # Load ada-002 embeddings from config path
    ada_path = settings.model.ada_embeddings.paths.train_embeddings
    if not os.path.exists(ada_path):
        raise FileNotFoundError(f"Ada embeddings not found at {ada_path}. Run `make ada-embed` first.")
    
    logger.info(f"Loading ada-002 embeddings from {ada_path}...")
    ada_embeddings = np.load(ada_path)
    logger.info(f"Ada embedding shape: {ada_embeddings.shape}")
    
    # Load or generate base ST embeddings from config path
    base_path = settings.model.sentence_transformer.paths.train_embeddings
    base_embeddings = load_or_generate_base_embeddings(df, str(base_path))
    logger.info(f"Base ST embedding shape: {base_embeddings.shape}")
    
    # Validate embeddings match data
    if len(ada_embeddings) != len(df):
        raise ValueError(f"Ada embeddings ({len(ada_embeddings)}) don't match data ({len(df)}). Regenerate with `make ada-embed`.")
    if len(base_embeddings) != len(df):
        raise ValueError(f"Base ST embeddings ({len(base_embeddings)}) don't match data ({len(df)}). Delete and regenerate.")
    
    # Compute full similarity matrices
    logger.info("Computing ada similarity matrix...")
    ada_sims = cosine_similarity(ada_embeddings)
    logger.info("Computing base ST similarity matrix...")
    base_sims = cosine_similarity(base_embeddings)
    
    # Compute percentile-based thresholds from the actual data
    # Exclude diagonal (self-similarity) for percentile calculation
    ada_sims_flat = ada_sims[np.triu_indices(n_papers, k=1)]  # Upper triangle, no diagonal
    ada_pos_threshold = np.percentile(ada_sims_flat, pos_percentile)
    ada_neg_threshold = np.percentile(ada_sims_flat, neg_percentile)
    
    logger.info(f"Ada similarity distribution: min={ada_sims_flat.min():.3f}, max={ada_sims_flat.max():.3f}, mean={ada_sims_flat.mean():.3f}")
    logger.info(f"Percentile thresholds: pos >= {ada_pos_threshold:.3f} ({pos_percentile}th), neg <= {ada_neg_threshold:.3f} ({neg_percentile}th)")
    
    # Pre-extract texts and IDs for efficiency
    texts = df['text'].tolist()
    ids = df['id'].tolist()
    
    hard_positives = []   # ada says similar, base says not
    agreement_pairs = []   # both say similar (for training stability)
    negative_pairs = []    # ada says NOT similar (CRITICAL for preventing collapse)
    seen_pairs = set()     # Track (min_idx, max_idx) to avoid duplicates
    
    logger.info(f"Mining pairs: checking top/bottom-{top_k} ada matches per paper...")
    logger.info(f"Hard pair gap threshold: {gap_threshold}")
    
    for i in tqdm(range(n_papers), desc="Processing papers"):
        # Get ada's similarities for this paper
        ada_scores = ada_sims[i].copy()
        ada_scores[i] = -1  # Exclude self
        
        # Get top-k indices by ada similarity (for positive pairs)
        top_k_indices = np.argsort(ada_scores)[::-1][:top_k]
        
        for j in top_k_indices:
            # Canonical ordering to avoid duplicates
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            ada_sim = ada_scores[j]
            base_sim = base_sims[i, j]
            gap = ada_sim - base_sim
            
            pair_data = {
                'idx1': int(i),
                'idx2': int(j),
                'id1': ids[i],
                'id2': ids[j],
                'text1': texts[i],
                'text2': texts[j],
                'ada_sim': float(ada_sim),
                'base_sim': float(base_sim),
                'gap': float(gap),
                'similarity_score': float(ada_sim),
            }
            
            # POSITIVE: ada says similar
            if ada_sim >= ada_pos_threshold:
                # Hard positive: base disagrees significantly
                if gap > gap_threshold:
                    hard_positives.append(pair_data)
                # Agreement: both think similar
                elif base_sim > 0.5:
                    agreement_pairs.append(pair_data)
        
        # Also sample some NEGATIVE pairs (ada says NOT similar)
        if include_negatives:
            # Get bottom-k (least similar according to ada)
            # Skip index 0 because ada_scores[i] = -1 would be at position 0
            bottom_k_indices = np.argsort(ada_scores)[1:top_k+1]  # Skip self (index 0 after sort)
            
            for j in bottom_k_indices:
                if j == i:  # Extra safety check - skip self
                    continue
                pair_key = (min(i, j), max(i, j))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                ada_sim = ada_scores[j]
                base_sim = base_sims[i, j]
                
                # Only include if ada says dissimilar
                if ada_sim < ada_neg_threshold:
                    negative_pairs.append({
                        'idx1': int(i),
                        'idx2': int(j),
                        'id1': ids[i],
                        'id2': ids[j],
                        'text1': texts[i],
                        'text2': texts[j],
                        'ada_sim': float(ada_sim),
                        'base_sim': float(base_sim),
                        'gap': float(ada_sim - base_sim),
                        'similarity_score': float(ada_sim),
                    })
    
    # Sort hard positives by gap (biggest disagreement first)
    hard_positives.sort(key=lambda x: x['gap'], reverse=True)
    # Sort negatives by ada_sim (lowest first)
    negative_pairs.sort(key=lambda x: x['ada_sim'])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PAIR MINING RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total pairs checked: {len(seen_pairs)}")
    logger.info(f"Hard Positives (ada high, base low): {len(hard_positives)}")
    logger.info(f"Agreement Pairs (both similar): {len(agreement_pairs)}")
    logger.info(f"Negative Pairs (ada says dissimilar): {len(negative_pairs)}")
    
    # Show examples
    if hard_positives:
        logger.info(f"\n--- Example Hard Positives (base ST misses these) ---")
        for p in hard_positives[:3]:
            logger.info(f"  Ada: {p['ada_sim']:.3f}, Base: {p['base_sim']:.3f}, Gap: {p['gap']:.3f}")
            logger.info(f"    Paper 1: {p['text1'][:80]}...")
            logger.info(f"    Paper 2: {p['text2'][:80]}...")
    
    if negative_pairs:
        logger.info(f"\n--- Example Negative Pairs (dissimilar papers) ---")
        for p in negative_pairs[:3]:
            logger.info(f"  Ada: {p['ada_sim']:.3f}, Base: {p['base_sim']:.3f}")
            logger.info(f"    Paper 1: {p['text1'][:80]}...")
            logger.info(f"    Paper 2: {p['text2'][:80]}...")
    
    # Create BALANCED training set: 50% positive, 50% negative
    # This is CRITICAL to prevent embedding space collapse!
    n_pos_available = len(hard_positives) + len(agreement_pairs)
    n_neg_available = len(negative_pairs)
    
    # Balance: equal positive and negative
    n_each = min(n_pos_available, n_neg_available, n_pairs // 2)
    
    # For positives: prioritize hard pairs, then agreement
    n_hard = min(len(hard_positives), int(n_each * 0.8))
    n_agree = min(len(agreement_pairs), n_each - n_hard)
    positive_final = hard_positives[:n_hard] + agreement_pairs[:n_agree]
    
    # For negatives: take lowest ada_sim pairs
    n_neg = min(len(negative_pairs), n_each)
    negative_final = negative_pairs[:n_neg]
    
    final_pairs = positive_final + negative_final
    random.shuffle(final_pairs)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL BALANCED TRAINING SET: {len(final_pairs)} pairs")
    logger.info(f"  Positive pairs: {len(positive_final)} (hard: {n_hard}, agreement: {n_agree})")
    logger.info(f"  Negative pairs: {len(negative_final)}")
    if len(final_pairs) < n_pairs:
        logger.warning(f"  WARNING: Only found {len(final_pairs)} pairs (requested {n_pairs})")
    logger.info(f"{'='*60}")
    
    # Score distribution stats
    if final_pairs:
        all_scores = [p['ada_sim'] for p in final_pairs]
        logger.info(f"\nAda similarity stats: min={min(all_scores):.3f}, max={max(all_scores):.3f}, mean={np.mean(all_scores):.3f}")
    
    # Create output DataFrame
    output_df = pd.DataFrame([
        {
            'text1': p['text1'],
            'text2': p['text2'],
            'similarity_score': p['similarity_score'],
            'id1': p['id1'],
            'id2': p['id2'],
            'ada_sim': p['ada_sim'],
            'base_sim': p['base_sim'],
            'gap': p['gap'],
        }
        for p in final_pairs
    ])
    
    # Save using path from config
    output_path = Path(settings.data.hard_pairs.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved training pairs to {output_path}")
    
    # Save statistics
    stats = {
        'n_papers': n_papers,
        'top_k_per_paper': top_k,
        'total_pairs_checked': len(seen_pairs),
        'hard_positives_found': len(hard_positives),
        'agreement_pairs_found': len(agreement_pairs),
        'negative_pairs_found': len(negative_pairs),
        'final_positive_pairs': len(positive_final),
        'final_negative_pairs': len(negative_final),
        'final_total': len(final_pairs),
        'requested_pairs': n_pairs,
        'thresholds': {
            'pos_percentile': pos_percentile,
            'neg_percentile': neg_percentile,
            'ada_pos_threshold': float(ada_pos_threshold),
            'ada_neg_threshold': float(ada_neg_threshold),
            'gap_threshold': gap_threshold,
        }
    }
    stats_path = output_path.parent / "hard_pairs_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")
    
    return output_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mine hard training pairs")
    parser.add_argument("--n-pairs", type=int, default=None, help="Target number of pairs (default from config)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k ada matches to check per paper")
    parser.add_argument("--pos-percentile", type=float, default=75, help="Percentile for positive threshold")
    parser.add_argument("--neg-percentile", type=float, default=25, help="Percentile for negative threshold")
    parser.add_argument("--gap-threshold", type=float, default=0.2, help="Min gap for hard positive")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    mine_hard_pairs(
        n_pairs=args.n_pairs,
        top_k=args.top_k,
        pos_percentile=args.pos_percentile,
        neg_percentile=args.neg_percentile,
        gap_threshold=args.gap_threshold,
        random_seed=args.seed,
    )
