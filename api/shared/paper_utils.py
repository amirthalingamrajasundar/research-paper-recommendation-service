"""
Utility functions for paper data handling.
"""
import math
from typing import List, Dict, Any, Optional
import pandas as pd

from .models import Paper, Version, Pagination


def parse_authors(authors_str: str) -> List[str]:
    """Parse authors string to list."""
    if not authors_str or pd.isna(authors_str):
        return []
    # Handle common author string formats
    # Remove LaTeX-style escapes
    authors_str = authors_str.replace("\\'", "'").replace("\\`", "`")
    # Split by comma or "and"
    authors = [a.strip() for a in authors_str.replace(" and ", ", ").split(",")]
    return [a for a in authors if a]


def parse_categories(categories_str: str) -> List[str]:
    """Parse categories string to list."""
    if not categories_str or pd.isna(categories_str):
        return []
    return categories_str.split()


def parse_versions(versions_data: Any) -> List[Version]:
    """Parse versions data to list of Version objects."""
    if not versions_data or pd.isna(versions_data):
        return []
    if isinstance(versions_data, str):
        import ast
        try:
            versions_data = ast.literal_eval(versions_data)
        except:
            return []
    if isinstance(versions_data, list):
        return [
            Version(
                version=v.get('version', 'v1'),
                created=v.get('created', '')
            )
            for v in versions_data if isinstance(v, dict)
        ]
    return []


def row_to_paper(row: pd.Series, similarity_score: Optional[float] = None) -> Paper:
    """Convert a DataFrame row to a Paper object."""
    paper_id = str(row.get('id', ''))
    
    return Paper(
        id=paper_id,
        title=str(row.get('title', '')).strip(),
        authors=parse_authors(row.get('authors', '')),
        abstract=str(row.get('abstract', '')).strip(),
        categories=parse_categories(row.get('categories', '')),
        submitter=row.get('submitter') if pd.notna(row.get('submitter')) else None,
        comments=row.get('comments') if pd.notna(row.get('comments')) else None,
        journal_ref=row.get('journal-ref') if pd.notna(row.get('journal-ref')) else None,
        doi=row.get('doi') if pd.notna(row.get('doi')) else None,
        versions=parse_versions(row.get('versions')),
        similarity_score=similarity_score,
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
        arxiv_url=f"https://arxiv.org/abs/{paper_id}"
    )


def create_pagination(
    total_results: int,
    page: int,
    limit: int
) -> Pagination:
    """Create pagination metadata."""
    total_pages = math.ceil(total_results / limit) if total_results > 0 else 1
    
    return Pagination(
        current_page=page,
        total_pages=total_pages,
        total_results=total_results,
        per_page=limit,
        has_next=page < total_pages,
        has_previous=page > 1
    )
