"""
API Gateway that routes requests to the appropriate backend service.
"""
import os
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# Service URLs (configurable via environment variables)
TFIDF_SERVICE_URL = os.getenv("TFIDF_SERVICE_URL", "http://localhost:8001")
ST_SERVICE_URL = os.getenv("ST_SERVICE_URL", "http://localhost:8002")
FT_SERVICE_URL = os.getenv("FT_SERVICE_URL", "http://localhost:8003")

# Model to service mapping
MODEL_SERVICES = {
    "tfidf": TFIDF_SERVICE_URL,
    "base_transformer": ST_SERVICE_URL,
    "fine_tuned_transformer": FT_SERVICE_URL,
}

# HTTP client
client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage HTTP client lifecycle."""
    global client
    client = httpx.AsyncClient(timeout=30.0)
    yield
    await client.aclose()


# FastAPI app
app = FastAPI(
    title="Scholar Stream API Gateway",
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


def get_service_url(model: str) -> str:
    """Get the service URL for a model."""
    if model not in MODEL_SERVICES:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": {
                    "code": "INVALID_MODEL",
                    "message": f"Invalid model. Must be one of: {', '.join(MODEL_SERVICES.keys())}"
                }
            }
        )
    return MODEL_SERVICES[model]


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "gateway"}


@app.get("/api/v1/scholar-stream/search")
async def search(
    q: Optional[str] = Query(None, description="Search query"),
    model: str = Query(..., description="Recommendation model"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(6, ge=1, le=50, description="Results per page")
):
    """Search papers - routes to appropriate backend service."""
    service_url = get_service_url(model)
    
    params = {"model": model, "page": page, "limit": limit}
    if q:
        params["q"] = q
    
    try:
        response = await client.get(
            f"{service_url}/scholar-stream/search",
            params=params
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": f"Backend service unavailable: {str(e)}"
                }
            }
        )


@app.get("/api/v1/scholar-stream/recommendations")
async def recommendations(
    paper_id: str = Query(..., description="Paper ID"),
    model: str = Query(..., description="Recommendation model"),
    limit: int = Query(3, ge=1, le=20, description="Number of recommendations")
):
    """Get recommendations - routes to appropriate backend service."""
    service_url = get_service_url(model)
    
    try:
        response = await client.get(
            f"{service_url}/scholar-stream/recommendations",
            params={"paper_id": paper_id, "model": model, "limit": limit}
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": f"Backend service unavailable: {str(e)}"
                }
            }
        )


@app.get("/api/v1/scholar-stream/paper/{paper_id}")
async def paper_detail(paper_id: str):
    """Get paper details - routes to tfidf service (any service has the same data)."""
    try:
        response = await client.get(
            f"{TFIDF_SERVICE_URL}/scholar-stream/paper/{paper_id}"
        )
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": f"Backend service unavailable: {str(e)}"
                }
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
