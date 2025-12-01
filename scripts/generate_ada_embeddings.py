"""
Generate embeddings using Azure OpenAI text-embedding-ada-002 model.

This script generates "teacher" embeddings that will be used to:
1. Find hard training pairs (where base ST disagrees with ada-002)
2. Provide target similarities for fine-tuning
"""
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaEmbeddingGenerator:
    """Generate embeddings using Azure OpenAI ada-002."""
    
    def __init__(self, batch_size: int = None, max_retries: int = None):
        ada_config = settings.model.ada_embeddings
        self.batch_size = batch_size or ada_config.batch_size
        self.max_retries = max_retries or ada_config.max_retries
        self.embeddings_model = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            api_version="2024-12-01-preview",
            model=os.getenv("EMBEDDING_MODEL_NAME"),
            chunk_size=self.batch_size,
        )
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        # Process in batches
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating ada embeddings", total=len(texts)//self.batch_size + 1)
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    batch_embeddings = self.embeddings_model.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Batch {i//self.batch_size} failed (attempt {attempt+1}): {e}")
                        asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"Batch {i//self.batch_size} failed after {self.max_retries} attempts")
                        # Fill with zeros for failed batch
                        all_embeddings.extend([[0.0] * 1536] * len(batch))
        
        return np.array(all_embeddings)


def generate_ada_embeddings(
    data_path: str = None,
    output_path: str = None,
    batch_size: int = None,
):
    """
    Generate ada-002 embeddings for papers.
    
    Args:
        data_path: Path to parquet file with papers (default: train_data from config)
        output_path: Path to save embeddings numpy file (default: from config)
        batch_size: Batch size for API calls (default: from config)
    """
    ada_config = settings.model.ada_embeddings
    data_path = data_path or str(settings.data.dataset.paths.train_data)
    output_path = output_path or str(ada_config.paths.train_embeddings)
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} papers")
    
    # Check for existing embeddings (for resuming)
    if os.path.exists(output_path):
        existing = np.load(output_path)
        if len(existing) == len(df):
            logger.info(f"Embeddings already exist at {output_path}")
            return existing
        else:
            logger.info(f"Found partial embeddings ({len(existing)}/{len(df)}), regenerating...")
    
    # Generate embeddings
    generator = AdaEmbeddingGenerator(batch_size=batch_size)
    texts = df['text'].tolist()
    
    embeddings = generator.generate_embeddings(texts)
    
    # Save
    np.save(output_path, embeddings)
    logger.info(f"Saved embeddings to {output_path}")
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    return embeddings


def generate_all_embeddings():
    """Generate ada embeddings for both train and holdout sets."""
    ada_paths = settings.model.ada_embeddings.paths
    data_paths = settings.data.dataset.paths
    
    # Train set
    logger.info("=" * 50)
    logger.info("Generating embeddings for TRAIN set...")
    logger.info("=" * 50)
    generate_ada_embeddings(
        data_path=str(data_paths.train_data),
        output_path=str(ada_paths.train_embeddings),
    )
    
    # Holdout set
    logger.info("=" * 50)
    logger.info("Generating embeddings for HOLDOUT set...")
    logger.info("=" * 50)
    generate_ada_embeddings(
        data_path=str(data_paths.holdout_data),
        output_path=str(ada_paths.holdout_embeddings),
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ada-002 embeddings")
    parser.add_argument("--train-only", action="store_true", help="Only generate for train set")
    parser.add_argument("--holdout-only", action="store_true", help="Only generate for holdout set")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for API calls (default from config)")
    
    args = parser.parse_args()
    
    ada_paths = settings.model.ada_embeddings.paths
    data_paths = settings.data.dataset.paths
    
    if args.train_only:
        generate_ada_embeddings(
            data_path=str(data_paths.train_data),
            output_path=str(ada_paths.train_embeddings),
            batch_size=args.batch_size,
        )
    elif args.holdout_only:
        generate_ada_embeddings(
            data_path=str(data_paths.holdout_data),
            output_path=str(ada_paths.holdout_embeddings),
            batch_size=args.batch_size,
        )
    else:
        generate_all_embeddings()
