"""
Script to train baseline models (TF-IDF and Sentence Transformer).
"""
import argparse
import logging
import os

import pandas as pd

from src.config import settings
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)
from src.models.tfidf.trainer import train_tfidf
from src.models.sentence_transformer.base_model import train_sentence_transformer
from src.models.sentence_transformer.finetuner import (
    finetune_sentence_transformer,
    generate_finetuned_embeddings
)


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument('--baseline', action='store_true', help='Train baseline models (TF-IDF + ST)')
    parser.add_argument('--tfidf', action='store_true', help='Train TF-IDF only')
    parser.add_argument('--st', action='store_true', help='Train Sentence Transformer only')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune Sentence Transformer')
    args = parser.parse_args()
    
    # Default to baseline if no specific flag
    if not any([args.baseline, args.tfidf, args.st, args.finetune]):
        args.baseline = True
    
    # Load data
    logger.info("Loading processed data...")
    df = pd.read_parquet(settings.data.dataset.paths.processed_data)
    logger.info(f"Loaded {len(df)} papers")
    
    if args.baseline or args.tfidf:
        logger.info("=" * 50)
        logger.info("Training TF-IDF model...")
        logger.info("=" * 50)
        tfidf_model = train_tfidf(df)
        
        # Quick test
        logger.info("TF-IDF Test - Top 3 recommendations for first paper:")
        recs = tfidf_model.get_recommendations_for_paper(0, top_k=3)
        for _, row in recs.iterrows():
            logger.info(f"  {row['rank']}. {row['title'][:50]}... (sim={row['similarity']:.3f})")
    
    if args.baseline or args.st:
        logger.info("=" * 50)
        logger.info("Training Sentence Transformer model...")
        logger.info("=" * 50)
        st_model = train_sentence_transformer(df)
        
        # Quick test
        logger.info("ST Test - Top 3 recommendations for first paper:")
        recs = st_model.get_recommendations_for_paper(0, top_k=3)
        for _, row in recs.iterrows():
            logger.info(f"  {row['rank']}. {row['title'][:50]}... (sim={row['similarity']:.3f})")
    
    if args.finetune:
        logger.info("=" * 50)
        logger.info("Fine-tuning Sentence Transformer...")
        logger.info("=" * 50)
        
        # Check if annotations exist
        annotations_path = settings.data.annotation.paths.output
        if not os.path.exists(annotations_path):
            logger.error(f"Annotations not found at {annotations_path}")
            logger.error("Run `make annotate` first to generate training data.")
            return
        
        # Fine-tune
        model = finetune_sentence_transformer(df)
        
        # Generate embeddings with fine-tuned model
        logger.info("Generating embeddings with fine-tuned model...")
        generate_finetuned_embeddings(model, df)
    
    logger.info("=" * 50)
    logger.info("Training complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
