"""
Generate embeddings for holdout set using all models.

This script generates embeddings for the 5K holdout papers using:
1. TF-IDF (fitted on train set)
2. Base Sentence Transformer (all-MiniLM-L6-v2)
3. Fine-tuned Sentence Transformer
4. Ada-002 (already generated separately)
"""
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_tfidf_embeddings(train_df: pd.DataFrame, holdout_df: pd.DataFrame):
    """Generate TF-IDF embeddings for holdout set."""
    tfidf_config = settings.model.tfidf
    vectorizer_path = Path(tfidf_config.paths.vectorizer)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if vectorizer exists, otherwise fit on train set
    if vectorizer_path.exists():
        logger.info(f"Loading existing TF-IDF vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        logger.info("Fitting TF-IDF vectorizer on training data...")
        vectorizer = TfidfVectorizer(
            max_features=tfidf_config.max_features,
            ngram_range=tuple(tfidf_config.ngram_range),
            min_df=tfidf_config.min_df,
            max_df=tfidf_config.max_df,
        )
        vectorizer.fit(train_df['text'].tolist())
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"Saved vectorizer to {vectorizer_path}")
    
    # Generate embeddings for holdout
    logger.info("Generating TF-IDF embeddings for holdout set...")
    holdout_embeddings = vectorizer.transform(holdout_df['text'].tolist()).toarray()
    
    holdout_emb_path = vectorizer_path.parent / "holdout_embeddings.npy"
    np.save(holdout_emb_path, holdout_embeddings)
    logger.info(f"Saved holdout TF-IDF embeddings to {holdout_emb_path}")
    logger.info(f"Shape: {holdout_embeddings.shape}")
    
    # Also generate for train set if not exists
    train_emb_path = vectorizer_path.parent / "train_embeddings.npy"
    if not train_emb_path.exists():
        logger.info("Generating TF-IDF embeddings for train set...")
        train_embeddings = vectorizer.transform(train_df['text'].tolist()).toarray()
        np.save(train_emb_path, train_embeddings)
        logger.info(f"Saved train TF-IDF embeddings to {train_emb_path}")
    
    return holdout_embeddings


def generate_base_st_embeddings(train_df: pd.DataFrame, holdout_df: pd.DataFrame):
    """Generate base Sentence Transformer embeddings."""
    st_config = settings.model.sentence_transformer
    st_paths = st_config.paths
    
    # Ensure output directory exists
    Path(st_paths.holdout_embeddings).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading base Sentence Transformer ({st_config.base_model})...")
    model = SentenceTransformer(st_config.base_model)
    
    # Generate holdout embeddings
    holdout_emb_path = Path(st_paths.holdout_embeddings)
    logger.info("Generating base ST embeddings for holdout set...")
    holdout_embeddings = model.encode(
        holdout_df['text'].tolist(),
        show_progress_bar=True,
        batch_size=st_config.batch_size,
    )
    np.save(holdout_emb_path, holdout_embeddings)
    logger.info(f"Saved holdout base ST embeddings to {holdout_emb_path}")
    logger.info(f"Shape: {holdout_embeddings.shape}")
    
    # Also generate for train set if not exists
    train_emb_path = Path(st_paths.train_embeddings)
    if not train_emb_path.exists():
        logger.info("Generating base ST embeddings for train set...")
        train_embeddings = model.encode(
            train_df['text'].tolist(),
            show_progress_bar=True,
            batch_size=st_config.batch_size,
        )
        np.save(train_emb_path, train_embeddings)
        logger.info(f"Saved train base ST embeddings to {train_emb_path}")
    
    return holdout_embeddings


def generate_finetuned_st_embeddings(holdout_df: pd.DataFrame):
    """Generate fine-tuned Sentence Transformer embeddings."""
    ft_config = settings.model.sentence_transformer.fine_tuning
    ft_paths = ft_config.paths
    
    model_path = Path(ft_paths.model)
    if not model_path.exists():
        logger.warning(f"Fine-tuned model not found at {model_path}. Run `make finetune` first.")
        return None
    
    logger.info(f"Loading fine-tuned Sentence Transformer from {model_path}...")
    model = SentenceTransformer(str(model_path))
    
    # Generate holdout embeddings
    holdout_emb_path = Path(ft_paths.holdout_embeddings)
    holdout_emb_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generating fine-tuned ST embeddings for holdout set...")
    holdout_embeddings = model.encode(
        holdout_df['text'].tolist(),
        show_progress_bar=True,
        batch_size=settings.model.sentence_transformer.batch_size,
    )
    np.save(holdout_emb_path, holdout_embeddings)
    logger.info(f"Saved holdout fine-tuned ST embeddings to {holdout_emb_path}")
    logger.info(f"Shape: {holdout_embeddings.shape}")
    
    return holdout_embeddings


def main():
    """Generate all embeddings for holdout evaluation."""
    
    # Load data using paths from config
    train_path = settings.data.dataset.paths.train_data
    holdout_path = settings.data.dataset.paths.holdout_data
    
    if not os.path.exists(train_path) or not os.path.exists(holdout_path):
        raise FileNotFoundError(
            "Train/holdout data not found. Run `make data` first."
        )
    
    logger.info(f"Loading train data from {train_path}...")
    train_df = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(train_df)} train papers")
    
    logger.info(f"Loading holdout data from {holdout_path}...")
    holdout_df = pd.read_parquet(holdout_path)
    logger.info(f"Loaded {len(holdout_df)} holdout papers")
    
    # Generate embeddings for each model
    logger.info("\n" + "="*60)
    logger.info("GENERATING TF-IDF EMBEDDINGS")
    logger.info("="*60)
    generate_tfidf_embeddings(train_df, holdout_df)
    
    logger.info("\n" + "="*60)
    logger.info("GENERATING BASE ST EMBEDDINGS")
    logger.info("="*60)
    generate_base_st_embeddings(train_df, holdout_df)
    
    logger.info("\n" + "="*60)
    logger.info("GENERATING FINE-TUNED ST EMBEDDINGS")
    logger.info("="*60)
    generate_finetuned_st_embeddings(holdout_df)
    
    logger.info("\n" + "="*60)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("="*60)
    logger.info("Note: Ada-002 embeddings should be generated separately using generate_ada_embeddings.py")


if __name__ == "__main__":
    main()
