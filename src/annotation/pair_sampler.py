"""
Sample diverse paper pairs for annotation.
1. Same category pairs (expect HIGH similarity)
2. Same author pairs (expect HIGH similarity)
3. Related category pairs (expect MEDIUM similarity)
4. Random pairs (expect LOW similarity)
"""
import logging
import random
from itertools import combinations

import pandas as pd

logger = logging.getLogger(__name__)


def sample_diverse_pairs(df, n_pairs=10000):
    """
    Sample pairs with expected diversity in similarity scores.
    Oversamples each bucket, then trims to exactly n_pairs.
    Returns list of (paper1_id, paper2_id) tuples.
    """
    # Oversample target per bucket (will trim at the end)
    oversample_factor = 3
    target_per_bucket = (n_pairs // 4) * oversample_factor
    
    # Work with a smaller sample for efficiency
    # Minimum 1000 papers to ensure category/author diversity, or 10x n_pairs
    sample_size = min(len(df), max(1000, n_pairs * 10))
    df_sample = df.sample(n=sample_size, random_state=42)
    logger.info(f"Working with {len(df_sample)} sampled papers (from {len(df)} total)")
    
    # 1. Same category pairs (expect HIGH similarity)
    logger.info("Sampling same-category pairs...")
    same_cat_pairs = []
    n_categories = df_sample["primary_category"].nunique()
    n_per_category = max(1, target_per_bucket // n_categories) + 5
    for cat, group in df_sample.groupby('primary_category'):
        if len(group) >= 2:
            ids = group['id'].tolist()
            cat_pairs = list(combinations(ids, 2))
            random.shuffle(cat_pairs)
            same_cat_pairs.extend(cat_pairs[:n_per_category])
    random.shuffle(same_cat_pairs)
    same_cat_pairs = same_cat_pairs[:target_per_bucket]
    
    # 2. Same author pairs (expect HIGH similarity)
    logger.info("Sampling same-author pairs...")
    same_author_pairs = []
    for author, group in df_sample.groupby(df_sample['authors'].str.split(',').str[0]):
        if len(group) >= 2:
            ids = group['id'].tolist()
            ap = list(combinations(ids, 2))
            same_author_pairs.extend(ap[:5])
    random.shuffle(same_author_pairs)
    same_author_pairs = same_author_pairs[:target_per_bucket]
    
    # 3. Related category pairs (expect MEDIUM similarity)
    logger.info("Sampling cross-category pairs...")
    categories = df_sample['primary_category'].unique()
    cross_cat_pairs = []
    attempts = 0
    max_attempts = target_per_bucket * 10
    while len(cross_cat_pairs) < target_per_bucket and attempts < max_attempts:
        attempts += 1
        cat1, cat2 = random.sample(list(categories), 2)
        # Check if categories share a prefix (e.g., cs.ML, cs.AI)
        if cat1.split('.')[0] == cat2.split('.')[0]:
            cat1_papers = df_sample[df_sample['primary_category'] == cat1]
            cat2_papers = df_sample[df_sample['primary_category'] == cat2]
            if len(cat1_papers) > 0 and len(cat2_papers) > 0:
                p1 = cat1_papers.sample(1)['id'].iloc[0]
                p2 = cat2_papers.sample(1)['id'].iloc[0]
                cross_cat_pairs.append((p1, p2))
    
    # 4. Random pairs (expect LOW similarity)
    logger.info("Sampling random pairs...")
    all_ids = df_sample['id'].tolist()
    random_pairs = []
    for _ in range(target_per_bucket):
        p1, p2 = random.sample(all_ids, 2)
        random_pairs.append((p1, p2))
    
    # Take balanced samples from each bucket (n_pairs // 4 each)
    n_per_final_bucket = n_pairs // 4
    remainder = n_pairs % 4
    
    final_same_cat = same_cat_pairs[:n_per_final_bucket]
    final_same_author = same_author_pairs[:n_per_final_bucket]
    final_cross_cat = cross_cat_pairs[:n_per_final_bucket]
    final_random = random_pairs[:n_per_final_bucket + remainder]  # Give remainder to random
    
    # Combine (keeping balance)
    all_pairs = final_same_cat + final_same_author + final_cross_cat + final_random
    
    # Remove duplicates while preserving order
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        pair_key = tuple(sorted(p))
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append(p)
    
    random.shuffle(unique_pairs)
    
    logger.info(f"Total pairs: {len(unique_pairs)} (requested: {n_pairs})")
    logger.info(f"  Same category: {len(final_same_cat)}, Same author: {len(final_same_author)}, Cross category: {len(final_cross_cat)}, Random: {len(final_random)}")
    
    return unique_pairs