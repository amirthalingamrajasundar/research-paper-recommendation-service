import json
import logging
import os
from pathlib import Path
import zipfile

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from src.config import settings

logger = logging.getLogger(__name__)

def download_from_kaggle(dataset_name, destination_dir):
    """
    Download the dataset from Kaggle.
    """
    dataset_file_name = dataset_name.split('/')[-1]
    file_path = Path(os.path.join(destination_dir, f"{dataset_file_name}.zip"))
    if not os.path.exists(file_path):
        kaggle_json_path = settings.data.kaggle.paths.credentials
        if os.path.exists(kaggle_json_path):
                logger.info(f"Kaggle config found at {kaggle_json_path}")
        os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_json_path.parent)
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading {dataset_name} from Kaggle to {destination_dir}")
        api.dataset_download_files(dataset_name, path=file_path.parent)
        logger.info("Download complete")
    else:
        logger.info(f"File {file_path} already exists")
    # Unzip the file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(file_path.parent)



def load_dataset(filepath=None, total_size=None, max_per_category=None):
    """
    Load a balanced subset by capping each category.
    Returns all papers as a single DataFrame.

    """
    filepath = filepath or settings.data.dataset.paths.raw_data
    total_size = total_size or settings.data.sampling.total_size
    max_per_category = max_per_category or settings.data.sampling.max_per_category
    
    if not os.path.exists(filepath):
        logger.warning(f"File {filepath} not found, downloading from Kaggle...")
        download_from_kaggle(settings.data.kaggle.dataset_name, filepath.parent)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found after download.")

    all_papers = []
    category_counts = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            paper = json.loads(line)
            primary_cat = paper['categories'].split()[0]
            
            # Cap each category at max_per_category
            if category_counts.get(primary_cat, 0) < max_per_category:
                # Add processed fields
                paper['primary_category'] = primary_cat
                paper['text'] = f"{paper['title']} {paper['abstract']}"
                
                all_papers.append(paper)
                category_counts[primary_cat] = category_counts.get(primary_cat, 0) + 1
            
            # Stop when we have enough papers
            if len(all_papers) >= total_size:
                break
    
    df = pd.DataFrame(all_papers)
    
    logger.info(f"Loaded {len(df)} papers")
    logger.info(f"Categories in subset: {len(category_counts)}")
    logger.debug(f"Papers per category stats:\n{pd.Series(category_counts).describe()}")
    
    return df