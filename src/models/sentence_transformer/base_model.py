"""
Sentence Transformer based recommendation model.
"""
import logging
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings

logger = logging.getLogger(__name__)


class SentenceTransformerModel:
    """Sentence Transformer based recommendation model."""
    
    def __init__(self, model_name: str = None):
        self.config = settings.model.sentence_transformer
        self.model_name = model_name or self.config.base_model
        self.name = f"SentenceTransformer ({self.model_name})"
        self.model = None
        self.embeddings = None
        self.df = None
    
    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def fit(self, df: pd.DataFrame, text_column: str = 'text'):
        """
        Generate embeddings for the corpus.
        
        Args:
            df: DataFrame with papers
            text_column: Column containing text to encode
        """
        self.df = df
        self._load_model()
        
        logger.info(f"Generating embeddings for {len(df)} documents...")
        logger.info(f"Model: {self.model_name}, batch_size: {self.config.batch_size}")
        
        self.embeddings = self.model.encode(
            df[text_column].tolist(),
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Embeddings shape: {self.embeddings.shape}")
        return self
    
    def encode(self, texts):
        """Encode new texts to embeddings."""
        self._load_model()
        return self.model.encode(texts, convert_to_numpy=True)
    
    def get_recommendations(self, query_vector, top_k: int = 10):
        """
        Get top-k recommendations for a query vector.
        
        Returns:
            DataFrame with recommendations
        """
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k-1:-1][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            paper = self.df.iloc[idx].to_dict()
            results.append({
                'rank': rank,
                'similarity': similarities[idx],
                'id': paper.get('id'),
                'title': paper.get('title'),
                'primary_category': paper.get('primary_category'),
                'paper': paper
            })
        return pd.DataFrame(results)
    
    def get_recommendations_for_paper(self, paper_idx: int, top_k: int = 10):
        """Get recommendations for a paper by index."""
        query_vector = self.embeddings[paper_idx]
        return self.get_recommendations(query_vector, top_k)
    
    def get_recommendations_for_text(self, query_text: str, top_k: int = 10):
        """Get recommendations for a query text."""
        query_vector = self.encode([query_text])[0]
        return self.get_recommendations(query_vector, top_k)
    
    def save(self, embeddings_path=None):
        """Save embeddings."""
        embeddings_path = embeddings_path or settings.model.sentence_transformer.paths.embeddings
        
        # Create directories
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(embeddings_path, self.embeddings)
        logger.info(f"Embeddings saved to {embeddings_path}")
    
    def load(self, df: pd.DataFrame = None, embeddings_path=None):
        """Load embeddings."""
        embeddings_path = embeddings_path or settings.model.sentence_transformer.paths.embeddings
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings not found at {embeddings_path}. Run fit() first.")
        
        logger.info(f"Loading embeddings from {embeddings_path}...")
        self.embeddings = np.load(embeddings_path)
        self.df = df
        self._load_model()
        
        logger.info(f"Loaded embeddings: {self.embeddings.shape}")
        return self


def train_sentence_transformer(df: pd.DataFrame) -> SentenceTransformerModel:
    """
    Train (generate embeddings) for sentence transformer model.
    
    Args:
        df: DataFrame with 'text' column
        
    Returns:
        Trained SentenceTransformerModel
    """
    model = SentenceTransformerModel()
    model.fit(df)
    model.save()
    return model


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # Quick test
    df = pd.read_parquet(settings.data.dataset.paths.processed_data)
    logger.info(f"Loaded {len(df)} papers")
    
    model = train_sentence_transformer(df)
    
    # Test recommendations
    logger.info("Test recommendations for first paper:")
    logger.info(f"Query: {df.iloc[0]['title']}")
    recs = model.get_recommendations_for_paper(0, top_k=5)
    for _, row in recs.iterrows():
        logger.info(f"  {row['rank']}. {row['title'][:60]}... (sim={row['similarity']:.3f})")
