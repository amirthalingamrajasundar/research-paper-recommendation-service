"""
Fine-tuned Sentence Transformer recommendation service.
"""
import os
import sys
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.shared.models import (
    SearchResponse, SearchData,
    RecommendationsResponse, RecommendationsData,
    PaperDetailResponse,
    ErrorResponse, ErrorDetail
)
from api.shared.paper_utils import row_to_paper, create_pagination


# Configuration
DATA_PATH = os.getenv("DATA_PATH", str(PROJECT_ROOT / "data/processed/index_100k.parquet"))
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models/finetuned_st/model"))
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", str(PROJECT_ROOT / "models/finetuned_st/embeddings.npy"))
MODEL_NAME = "fine_tuned_transformer"


class FinetunedService:
    """Fine-tuned Sentence Transformer based recommendation service."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.id_to_idx: dict = {}
    
    def load(self):
        """Load data and model."""
        print(f"Loading data from {DATA_PATH}...")
        self.df = pd.read_parquet(DATA_PATH)
        print(f"Loaded {len(self.df)} papers")
        
        # Build ID to index mapping
        self.id_to_idx = {str(row['id']): idx for idx, row in self.df.iterrows()}
        
        print(f"Loading fine-tuned model from {MODEL_PATH}...")
        self.model = SentenceTransformer(MODEL_PATH)
        
        print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
        self.embeddings = np.load(EMBEDDINGS_PATH)
        print(f"Embeddings shape: {self.embeddings.shape}")
    
    def search(self, query: str, page: int = 1, limit: int = 6) -> tuple:
        """
        Search papers by query text.
        Returns (papers, total_count)
        """
        if not query or query.strip() == "":
            # Return recent papers if no query
            total = len(self.df)
            start = (page - 1) * limit
            end = start + limit
            results = self.df.iloc[start:end]
            return [(row, None) for _, row in results.iterrows()], total
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top results
        top_indices = similarities.argsort()[::-1]
        total = len(top_indices)
        
        # Paginate
        start = (page - 1) * limit
        end = start + limit
        page_indices = top_indices[start:end]
        
        results = [
            (self.df.iloc[idx], similarities[idx])
            for idx in page_indices
        ]
        
        return results, total
    
    def get_recommendations(self, paper_id: str, limit: int = 3) -> list:
        """Get recommendations for a paper."""
        if paper_id not in self.id_to_idx:
            return None
        
        idx = self.id_to_idx[paper_id]
        query_embedding = self.embeddings[idx]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top results (excluding self)
        top_indices = similarities.argsort()[::-1]
        
        results = []
        for i in top_indices:
            if i != idx:
                results.append((self.df.iloc[i], similarities[i]))
            if len(results) >= limit:
                break
        
        return results
    
    def get_paper(self, paper_id: str) -> Optional[pd.Series]:
        """Get paper by ID."""
        if paper_id not in self.id_to_idx:
            return None
        return self.df.iloc[self.id_to_idx[paper_id]]


# Global service instance
service = FinetunedService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    service.load()
    yield


# FastAPI app
app = FastAPI(
    title="Fine-tuned Recommendation Service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": MODEL_NAME}


@app.get("/scholar-stream/search", response_model=SearchResponse)
async def search(
    q: Optional[str] = Query(None, description="Search query"),
    model: str = Query(..., description="Model name (ignored)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(6, ge=1, le=50, description="Results per page")
):
    """Search papers by query."""
    results, total = service.search(q or "", page, limit)
    
    papers = [
        row_to_paper(row, similarity_score=float(score) if score is not None else None)
        for row, score in results
    ]
    
    return SearchResponse(
        success=True,
        data=SearchData(
            papers=papers,
            pagination=create_pagination(total, page, limit)
        )
    )


@app.get("/scholar-stream/recommendations", response_model=RecommendationsResponse)
async def recommendations(
    paper_id: str = Query(..., description="Paper ID"),
    model: str = Query(..., description="Model name (ignored)"),
    limit: int = Query(3, ge=1, le=20, description="Number of recommendations")
):
    """Get recommendations for a paper."""
    results = service.get_recommendations(paper_id, limit)
    
    if results is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "PAPER_NOT_FOUND", "message": f"Paper {paper_id} not found"}
        )
    
    papers = [
        row_to_paper(row, similarity_score=float(score))
        for row, score in results
    ]
    
    return RecommendationsResponse(
        success=True,
        data=RecommendationsData(
            paper_id=paper_id,
            model_used=MODEL_NAME,
            recommendations=papers
        )
    )


@app.get("/scholar-stream/paper/{paper_id}", response_model=PaperDetailResponse)
async def paper_detail(paper_id: str):
    """Get paper details."""
    row = service.get_paper(paper_id)
    
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "PAPER_NOT_FOUND", "message": f"Paper {paper_id} not found"}
        )
    
    return PaperDetailResponse(
        success=True,
        data=row_to_paper(row)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
