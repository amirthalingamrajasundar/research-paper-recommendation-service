"""
TF-IDF model trainer and recommender.
"""
import logging
import os

import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import settings

logger = logging.getLogger(__name__)


class TFIDFModel:
    """TF-IDF based recommendation model."""
    
    def __init__(self):
        self.name = "TF-IDF"
        self.config = settings.model.tfidf
        self.vectorizer = None
        self.embeddings = None
        self.df = None  # Reference to paper dataframe
    
    def fit(self, df: pd.DataFrame, text_column: str = 'text'):
        """
        Fit the TF-IDF vectorizer on the corpus.
        
        Args:
            df: DataFrame with papers
            text_column: Column containing text to vectorize
        """
        self.df = df
        
        logger.info(f"Fitting TF-IDF on {len(df)} documents...")
        logger.info(f"Config: max_features={self.config.max_features}, "
                    f"ngram_range={tuple(self.config.ngram_range)}, "
                    f"min_df={self.config.min_df}, max_df={self.config.max_df}")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=tuple(self.config.ngram_range),
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            stop_words='english'
        )
        
        self.embeddings = self.vectorizer.fit_transform(df[text_column])
        
        logger.info(f"TF-IDF Matrix Shape: {self.embeddings.shape}")
        logger.info(f"Matrix sparsity: {(1 - self.embeddings.nnz / (self.embeddings.shape[0] * self.embeddings.shape[1])) * 100:.2f}%")
        
        return self
    
    def transform(self, texts):
        """Transform new texts to TF-IDF vectors."""
        if self.vectorizer is None:
            raise ValueError("Model not fitted. Call fit() first or load().")
        return self.vectorizer.transform(texts)
    
    def get_recommendations(self, query_vector, top_k: int = 10):
        """
        Get top-k recommendations for a query vector.
        
        Returns:
            DataFrame with recommendations
        """
        similarities = cosine_similarity(query_vector, self.embeddings).flatten()
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
        query_vector = self.transform([query_text])
        return self.get_recommendations(query_vector, top_k)
    
    def save(self):
        """Save vectorizer and embeddings."""
        vectorizer_path = settings.model.tfidf.paths.vectorizer
        embeddings_path = settings.model.tfidf.paths.embeddings
        
        # Create directories
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        
        dump(self.vectorizer, vectorizer_path)
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        dump(self.embeddings, embeddings_path)
        logger.info(f"Embeddings saved to {embeddings_path}")
    
    def load(self, df: pd.DataFrame = None):
        """Load vectorizer and embeddings."""
        vectorizer_path = settings.model.tfidf.paths.vectorizer
        embeddings_path = settings.model.tfidf.paths.embeddings
        
        if not os.path.exists(vectorizer_path) or not os.path.exists(embeddings_path):
            raise FileNotFoundError("Model files not found. Run fit() first.")
        
        logger.info("Loading TF-IDF model...")
        self.vectorizer = load(vectorizer_path)
        self.embeddings = load(embeddings_path)
        self.df = df
        
        logger.info(f"Loaded TF-IDF matrix: {self.embeddings.shape}")
        return self


def train_tfidf(df: pd.DataFrame) -> TFIDFModel:
    """
    Train TF-IDF model on the dataset.
    
    Args:
        df: DataFrame with 'text' column
        
    Returns:
        Trained TFIDFModel
    """
    model = TFIDFModel()
    model.fit(df)
    model.save()
    return model


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # Quick test
    df = pd.read_parquet(settings.data.dataset.paths.processed_data)
    logger.info(f"Loaded {len(df)} papers")
    
    model = train_tfidf(df)
    
    # Test recommendations
    logger.info("Test recommendations for first paper:")
    logger.info(f"Query: {df.iloc[0]['title']}")
    recs = model.get_recommendations_for_paper(0, top_k=5)
    for _, row in recs.iterrows():
        logger.info(f"  {row['rank']}. {row['title'][:60]}... (sim={row['similarity']:.3f})")
