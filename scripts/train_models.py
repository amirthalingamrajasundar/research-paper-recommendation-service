"""
Train recommendation models.

Usage:
    python -m scripts.train_models --baseline  # Train TF-IDF + base ST
    python -m scripts.train_models --finetune  # Fine-tune ST on hard pairs
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument('--baseline', action='store_true', help='Train baseline models (TF-IDF + base ST)')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune Sentence Transformer on hard pairs')
    args = parser.parse_args()
    
    # Default to baseline if no specific flag
    if not args.baseline and not args.finetune:
        args.baseline = True
    
    # Use train set for training (from config)
    train_path = Path(settings.data.dataset.paths.train_data)
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        logger.error("Run `make data` first to prepare the dataset.")
        return
    
    logger.info(f"Loading training data from {train_path}...")
    df = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(df)} papers")
    
    if args.baseline:
        from src.models.tfidf.trainer import train_tfidf
        from src.models.sentence_transformer.base_model import train_sentence_transformer
        
        logger.info("=" * 50)
        logger.info("Training TF-IDF model...")
        logger.info("=" * 50)
        tfidf_model = train_tfidf(df)
        
        logger.info(f"TF-IDF model saved to: {settings.model.tfidf.paths.vectorizer}")
        
        # Quick test
        logger.info("TF-IDF Test - Top 3 recommendations for first paper:")
        recs = tfidf_model.get_recommendations_for_paper(0, top_k=3)
        for _, row in recs.iterrows():
            logger.info(f"  {row['rank']}. {row['title'][:50]}... (sim={row['similarity']:.3f})")
        
        logger.info("=" * 50)
        logger.info("Training base Sentence Transformer model...")
        logger.info("=" * 50)
        st_model = train_sentence_transformer(df)
        
        logger.info(f"Base ST embeddings saved to: {settings.model.sentence_transformer.paths.embeddings}")
        
        # Quick test
        logger.info("Base ST Test - Top 3 recommendations for first paper:")
        recs = st_model.get_recommendations_for_paper(0, top_k=3)
        for _, row in recs.iterrows():
            logger.info(f"  {row['rank']}. {row['title'][:50]}... (sim={row['similarity']:.3f})")
    
    if args.finetune:
        from src.models.sentence_transformer.finetuner import (
            finetune_sentence_transformer,
            generate_finetuned_embeddings
        )
        
        logger.info("=" * 50)
        logger.info("Fine-tuning Sentence Transformer on hard pairs...")
        logger.info("=" * 50)
        
        # Use hard pairs from config
        annotations_path = Path(settings.data.hard_pairs.output)
        if not annotations_path.exists():
            logger.error(f"Hard pairs not found at {annotations_path}")
            logger.error("Run `make mine-pairs` first to generate training pairs.")
            return
        
        logger.info(f"Using hard pairs from: {annotations_path}")
        
        # Fine-tune
        model = finetune_sentence_transformer(
            annotations_path=str(annotations_path),
        )
        
        # Generate embeddings for train set
        logger.info("Generating embeddings with fine-tuned model...")
        generate_finetuned_embeddings(model, df)
        logger.info(f"Fine-tuned model saved to: {settings.model.sentence_transformer.fine_tuning.paths.model}")
        logger.info(f"Fine-tuned embeddings saved to: {settings.model.sentence_transformer.fine_tuning.paths.embeddings}")
    
    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
