"""
Prepare focused dataset for fine-tuning experiments.

Creates:
- 25K training papers (cs.AI)
- 5K holdout papers (cs.AI) for evaluation
"""
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_from_kaggle(dataset_name: str, destination_dir: Path):
    """
    Download the dataset from Kaggle and unzip it.
    
    Args:
        dataset_name: Kaggle dataset name (e.g., "Cornell-University/arxiv")
        destination_dir: Directory to save the downloaded file
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_file_name = dataset_name.split('/')[-1]
    zip_path = destination_dir / f"{dataset_file_name}.zip"
    
    if not zip_path.exists():
        # Set up Kaggle credentials
        kaggle_json_path = settings.data.kaggle.paths.credentials
        if os.path.exists(kaggle_json_path):
            logger.info(f"Kaggle config found at {kaggle_json_path}")
            os.environ['KAGGLE_CONFIG_DIR'] = str(Path(kaggle_json_path).parent)
        
        # Download
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading {dataset_name} from Kaggle to {destination_dir}...")
        api.dataset_download_files(dataset_name, path=str(destination_dir))
        logger.info("Download complete")
    else:
        logger.info(f"Zip file already exists at {zip_path}")
    
    # Unzip
    if zip_path.exists():
        logger.info(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
        logger.info("Unzip complete")


def is_clean_text(text: str) -> bool:
    """Check if text contains only printable ASCII characters."""
    if not text:
        return False
    try:
        return all(ord(c) < 128 for c in text)
    except:
        return False


def prepare_focused_data(
    total_size: int = None,
    holdout_size: int = None,
    category: str = None,
    random_seed: int = 42
):
    """
    Load cs.AI papers and split into train/holdout.
    
    Args:
        total_size: Total number of papers to load (default from config)
        holdout_size: Number of papers for holdout evaluation (default from config)
        category: ArXiv category to filter (default from config)
        random_seed: Random seed for reproducibility
    """
    # Load defaults from config
    prep_config = settings.data.preparation
    total_size = total_size or prep_config.total_size
    holdout_size = holdout_size or prep_config.holdout_size
    category = category or prep_config.category
    
    raw_path = Path(settings.data.dataset.paths.raw_data)
    train_path = settings.data.dataset.paths.train_data
    holdout_path = settings.data.dataset.paths.holdout_data
    processed_path = settings.data.dataset.paths.processed_data
    
    # Ensure output directories exist
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Download from Kaggle if raw file doesn't exist
    if not raw_path.exists():
        logger.warning(f"Raw data not found at {raw_path}")
        logger.info("Downloading from Kaggle...")
        download_from_kaggle(
            settings.data.kaggle.dataset_name,
            raw_path.parent
        )
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path} after download attempt")
    
    logger.info(f"Loading {category} papers from {raw_path}...")
    
    papers = []
    with open(raw_path, 'r') as f:
        for line in f:
            paper = json.loads(line)
            primary_cat = paper['categories'].split()[0]
            
            if primary_cat == category:
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if not is_clean_text(title) or not is_clean_text(abstract):
                    continue
                
                paper['primary_category'] = primary_cat
                paper['text'] = f"{title} {abstract}"
                papers.append(paper)
            
            if len(papers) >= total_size:
                break
    
    df = pd.DataFrame(papers)
    logger.info(f"Loaded {len(df)} {category} papers")
    
    if len(df) < total_size:
        logger.warning(f"Only found {len(df)} papers (requested {total_size})")
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    train_size = len(df) - holdout_size
    df_train = df.iloc[:train_size].reset_index(drop=True)
    df_holdout = df.iloc[train_size:].reset_index(drop=True)
    
    logger.info(f"Split: {len(df_train)} train, {len(df_holdout)} holdout")
    
    # Save using paths from config
    df_train.to_parquet(train_path, index=False)
    df_holdout.to_parquet(holdout_path, index=False)
    df.to_parquet(processed_path, index=False)
    
    logger.info(f"Saved train data to {train_path}")
    logger.info(f"Saved holdout data to {holdout_path}")
    logger.info(f"Saved combined data to {processed_path}")
    
    return df_train, df_holdout


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare focused dataset")
    parser.add_argument("--total-size", type=int, default=None, help="Total papers to load (default from config)")
    parser.add_argument("--holdout-size", type=int, default=None, help="Holdout set size (default from config)")
    parser.add_argument("--category", type=str, default=None, help="ArXiv category (default from config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    prepare_focused_data(
        total_size=args.total_size,
        holdout_size=args.holdout_size,
        category=args.category,
        random_seed=args.seed
    )
