# Scholar Stream API Specification

## Base URL

```
http://localhost:8000/api/v1
```

**Production URL:** (after Cloud Run deployment)
```
https://scholar-stream-gateway-<project-id>.run.app/api/v1
```

## Authentication

None required (public API)

---

## Models Available

| Model ID | Description | Performance | Use Case |
|----------|-------------|-------------|----------|
| `tfidf` | TF-IDF baseline model | Fastest (~50ms) | Quick keyword matching |
| `base_transformer` | Sentence Transformer (all-MiniLM-L6-v2) | Medium (~100ms) | Semantic similarity |
| `fine_tuned_transformer` | Fine-tuned on domain data | Medium (~100ms) | Best quality recommendations |

---

## Endpoints

### 1. Health Check

```http
GET /health
```

#### Response

```json
{
  "status": "healthy",
  "service": "gateway"
}
```

---

### 2. Search Papers

```http
GET /scholar-stream/search
```

#### Query Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `q` | string | No | - | - | Search query (natural language) |
| `model` | string | **Yes** | - | `tfidf`, `base_transformer`, `fine_tuned_transformer` | Recommendation model |
| `page` | integer | No | 1 | ≥ 1 | Page number |
| `limit` | integer | No | 6 | 1-50 | Results per page |

#### Example Request

```bash
curl "http://localhost:8000/api/v1/scholar-stream/search?q=machine+learning&model=tfidf&page=1&limit=6"
```

#### Success Response (200)

```json
{
  "success": true,
  "data": {
    "papers": [
      {
        "id": "0704.0001",
        "title": "Calculation of prompt diphoton production cross sections",
        "authors": ["C. Balazs", "E. L. Berger", "P. M. Nadolsky", "C.-P. Yuan"],
        "abstract": "A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders...",
        "categories": ["hep-ph"],
        "submitter": "Pavel Nadolsky",
        "comments": "37 pages, 15 figures; published version",
        "journal_ref": "Phys.Rev.D76:013009,2007",
        "doi": "10.1103/PhysRevD.76.013009",
        "versions": [
          {"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"},
          {"version": "v2", "created": "Tue, 24 Jul 2007 20:10:27 GMT"}
        ],
        "similarity_score": 0.8923,
        "pdf_url": "https://arxiv.org/pdf/0704.0001",
        "arxiv_url": "https://arxiv.org/abs/0704.0001"
      }
    ],
    "pagination": {
      "current_page": 1,
      "total_pages": 15,
      "total_results": 89,
      "per_page": 6,
      "has_next": true,
      "has_previous": false
    }
  }
}
```

---

### 3. Get Recommendations

```http
GET /scholar-stream/recommendations
```

#### Query Parameters

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `paper_id` | string | **Yes** | - | Valid ArXiv ID (e.g., `0704.0001`) | Source paper ID |
| `model` | string | **Yes** | - | `tfidf`, `base_transformer`, `fine_tuned_transformer` | Recommendation model |
| `limit` | integer | No | 3 | 1-20 | Number of recommendations |

#### Example Request

```bash
curl "http://localhost:8000/api/v1/scholar-stream/recommendations?paper_id=0704.0001&model=tfidf&limit=5"
```

#### Success Response (200)

```json
{
  "success": true,
  "data": {
    "paper_id": "0704.0001",
    "model_used": "tfidf",
    "recommendations": [
      {
        "id": "0704.0123",
        "title": "Similar Paper Title",
        "authors": ["Author Name"],
        "abstract": "Paper abstract text...",
        "categories": ["hep-ph", "hep-ex"],
        "submitter": null,
        "comments": null,
        "journal_ref": null,
        "doi": null,
        "versions": [],
        "similarity_score": 0.7845,
        "pdf_url": "https://arxiv.org/pdf/0704.0123",
        "arxiv_url": "https://arxiv.org/abs/0704.0123"
      }
    ]
  }
}
```

---

### 4. Get Paper Details

```http
GET /scholar-stream/paper/{paper_id}
```

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `paper_id` | string | **Yes** | ArXiv paper ID (e.g., `0704.0001`) |

#### Example Request

```bash
curl "http://localhost:8000/api/v1/scholar-stream/paper/0704.0001"
```

#### Success Response (200)

```json
{
  "success": true,
  "data": {
    "id": "0704.0001",
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "abstract": "Full abstract text...",
    "categories": ["hep-ph"],
    "submitter": "Author Name",
    "comments": "37 pages, 15 figures",
    "journal_ref": "Phys.Rev.D76:013009,2007",
    "doi": "10.1103/PhysRevD.76.013009",
    "versions": [
      {"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"}
    ],
    "similarity_score": null,
    "pdf_url": "https://arxiv.org/pdf/0704.0001",
    "arxiv_url": "https://arxiv.org/abs/0704.0001"
  }
}
```

---

## Error Responses

### Invalid Model (400)

```json
{
  "success": false,
  "error": {
    "code": "INVALID_MODEL",
    "message": "Invalid model. Must be one of: tfidf, base_transformer, fine_tuned_transformer"
  }
}
```

### Paper Not Found (404)

```json
{
  "success": false,
  "error": {
    "code": "NOT_FOUND",
    "message": "Paper not found: 9999.9999"
  }
}
```

### Validation Error (422)

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["query", "model"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

### Service Unavailable (503)

```json
{
  "success": false,
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "Backend service unavailable: Connection refused"
  }
}
```

---

## Data Types

### Paper Object

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| `id` | string | No | ArXiv paper ID (e.g., `0704.0001`) |
| `title` | string | No | Paper title |
| `authors` | string[] | No | List of author names |
| `abstract` | string | No | Paper abstract |
| `categories` | string[] | No | ArXiv categories (e.g., `["cs.LG", "cs.AI"]`) |
| `submitter` | string | Yes | Submitter name |
| `comments` | string | Yes | Author comments (pages, figures, etc.) |
| `journal_ref` | string | Yes | Journal reference |
| `doi` | string | Yes | DOI identifier |
| `versions` | Version[] | No | Version history (may be empty) |
| `similarity_score` | float | Yes | Similarity score (0.0-1.0), null for detail view |
| `pdf_url` | string | Yes | Direct PDF link |
| `arxiv_url` | string | Yes | ArXiv abstract page link |

### Version Object

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Version identifier (e.g., `"v1"`, `"v2"`) |
| `created` | string | Creation timestamp (RFC 2822 format) |

### Pagination Object

| Field | Type | Description |
|-------|------|-------------|
| `current_page` | integer | Current page number (1-indexed) |
| `total_pages` | integer | Total number of pages |
| `total_results` | integer | Total matching results |
| `per_page` | integer | Results per page |
| `has_next` | boolean | More pages available |
| `has_previous` | boolean | Previous pages available |

---

## Frontend Integration

### TypeScript Types

```typescript
// Core Types
interface Version {
  version: string;
  created: string;
}

interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  categories: string[];
  submitter: string | null;
  comments: string | null;
  journal_ref: string | null;
  doi: string | null;
  versions: Version[];
  similarity_score: number | null;
  pdf_url: string | null;
  arxiv_url: string | null;
}

interface Pagination {
  current_page: number;
  total_pages: number;
  total_results: number;
  per_page: number;
  has_next: boolean;
  has_previous: boolean;
}

// Response Types
interface SearchResponse {
  success: true;
  data: {
    papers: Paper[];
    pagination: Pagination;
  };
}

interface RecommendationsResponse {
  success: true;
  data: {
    paper_id: string;
    model_used: string;
    recommendations: Paper[];
  };
}

interface PaperDetailResponse {
  success: true;
  data: Paper;
}

interface ErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
  };
}

type ModelType = 'tfidf' | 'base_transformer' | 'fine_tuned_transformer';
```

### API Client Example

```typescript
const API_BASE = 'http://localhost:8000/api/v1';

class ScholarStreamAPI {
  async search(
    query: string,
    model: ModelType,
    page = 1,
    limit = 6
  ): Promise<SearchResponse | ErrorResponse> {
    const params = new URLSearchParams({
      q: query,
      model,
      page: String(page),
      limit: String(limit),
    });
    const response = await fetch(`${API_BASE}/scholar-stream/search?${params}`);
    return response.json();
  }

  async getRecommendations(
    paperId: string,
    model: ModelType,
    limit = 3
  ): Promise<RecommendationsResponse | ErrorResponse> {
    const params = new URLSearchParams({
      paper_id: paperId,
      model,
      limit: String(limit),
    });
    const response = await fetch(`${API_BASE}/scholar-stream/recommendations?${params}`);
    return response.json();
  }

  async getPaperDetails(paperId: string): Promise<PaperDetailResponse | ErrorResponse> {
    const response = await fetch(`${API_BASE}/scholar-stream/paper/${paperId}`);
    return response.json();
  }
}

// Usage
const api = new ScholarStreamAPI();

// Search
const searchResults = await api.search('machine learning', 'tfidf', 1, 10);
if (searchResults.success) {
  console.log(searchResults.data.papers);
  console.log(searchResults.data.pagination);
}

// Recommendations
const recs = await api.getRecommendations('0704.0001', 'base_transformer', 5);
if (recs.success) {
  console.log(recs.data.recommendations);
}

// Paper details
const paper = await api.getPaperDetails('0704.0001');
if (paper.success) {
  console.log(paper.data.title);
}
```

### React Hook Example

```typescript
import { useState, useEffect } from 'react';

function usePaperSearch(query: string, model: ModelType, page: number) {
  const [data, setData] = useState<SearchResponse['data'] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!query) return;

    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const api = new ScholarStreamAPI();
        const result = await api.search(query, model, page);
        if (result.success) {
          setData(result.data);
        } else {
          setError(result.error.message);
        }
      } catch (e) {
        setError('Network error');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [query, model, page]);

  return { data, loading, error };
}
```

---

## Notes for Frontend Development

### Display Guidelines

1. **Similarity Scores**: Display as percentage (`Math.round(score * 100)%`)
2. **Categories**: First category is primary; consider color-coding by field
3. **Authors**: Join with commas; truncate if > 5 authors with "+ N more"
4. **Abstract**: Truncate to ~200 chars in list view; full in detail view

### UX Recommendations

1. **Model Selection**: Dropdown or tabs to switch models; show comparison
2. **Pagination**: Use `has_next`/`has_previous` for button states
3. **Loading States**: Expect 1-3 seconds on cold starts
4. **Error Handling**: Always check `success` field before accessing `data`
5. **Empty States**: Handle no results gracefully

### CORS

API allows all origins (`*`). No special headers required.

### Rate Limits

No rate limits currently enforced. Be reasonable with requests.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-11 | Initial release |
