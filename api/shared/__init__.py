"""Shared API components."""
from .models import (
    Paper, Version, Pagination,
    SearchResponse, SearchData,
    RecommendationsResponse, RecommendationsData,
    PaperDetailResponse,
    ErrorResponse, ErrorDetail
)

__all__ = [
    'Paper', 'Version', 'Pagination',
    'SearchResponse', 'SearchData',
    'RecommendationsResponse', 'RecommendationsData',
    'PaperDetailResponse',
    'ErrorResponse', 'ErrorDetail'
]
