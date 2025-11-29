"""
Script to prepare a balanced dataset for training.
Downloads the dataset from Kaggle if needed.
Saves the dataset as a parquet file.
"""
import logging

from src.preprocessing.data_loader import load_dataset
from src.config import settings
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    
    df = load_dataset()
    output_path = settings.data.dataset.paths.processed_data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()