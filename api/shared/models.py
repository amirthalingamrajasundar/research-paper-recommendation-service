"""
Pydantic models for API responses.
"""
from typing import List, Optional
from pydantic import BaseModel


class Version(BaseModel):
    """Paper version info."""
    version: str
    created: str


class Paper(BaseModel):
    """Paper model for API responses."""
    id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    submitter: Optional[str] = None
    comments: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    versions: List[Version] = []
    similarity_score: Optional[float] = None
    pdf_url: Optional[str] = None
    arxiv_url: Optional[str] = None


class Pagination(BaseModel):
    """Pagination metadata."""
    current_page: int
    total_pages: int
    total_results: int
    per_page: int
    has_next: bool
    has_previous: bool


class SearchData(BaseModel):
    """Search response data."""
    papers: List[Paper]
    pagination: Pagination


class SearchResponse(BaseModel):
    """Search endpoint response."""
    success: bool = True
    data: SearchData


class RecommendationsData(BaseModel):
    """Recommendations response data."""
    paper_id: str
    model_used: str
    recommendations: List[Paper]


class RecommendationsResponse(BaseModel):
    """Recommendations endpoint response."""
    success: bool = True
    data: RecommendationsData


class PaperDetailResponse(BaseModel):
    """Paper detail endpoint response."""
    success: bool = True
    data: Paper


class ErrorDetail(BaseModel):
    """Error detail."""
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: ErrorDetail
