# Evaluation Procedure and Metrics

This document details the evaluation methodology used to assess the performance of recommendation models in Scholar Stream.

## Table of Contents

1. [Evaluation Overview](#evaluation-overview)
2. [Ground Truth: Ada-002 as Reference](#ground-truth-ada-002-as-reference)
3. [Evaluation Procedure](#evaluation-procedure)
4. [Metrics Explained](#metrics-explained)
5. [Training vs Evaluation Metrics](#training-vs-evaluation-metrics)
6. [Running Evaluations](#running-evaluations)

---

## Evaluation Overview

We evaluate models by comparing their recommendations against Ada-002's recommendations (ground truth). The core question:

> **"How well do the model's top-k recommendations match Ada-002's top-k recommendations?"**

### Why Ada-002 as Ground Truth?

- Ada-002 is a powerful embedding model trained on massive diverse data
- It captures nuanced semantic relationships between papers
- Using it as ground truth avoids expensive human annotations
- Our goal is to distill Ada-002's knowledge into a smaller, deployable model

---

## Ground Truth: Ada-002 as Reference

For each query paper, we define the "correct" recommendations as Ada-002's top-k most similar papers.

```python
# Get Ada-002's ranking for paper i
ada_similarities = cosine_similarity(ada_embeddings)
ada_ranking = get_top_k_indices(ada_similarities, paper_idx, k=10)
# Returns: [paper_7, paper_23, paper_156, ...] (top 10 most similar)
```

This becomes our ground truth: if Ada-002 says papers 7, 23, and 156 are most similar to paper i, a good model should also rank these papers highly.

---

## Evaluation Procedure

### Step 1: Load Embeddings

```python
# Load pre-computed embeddings for holdout set (5,000 papers)
ada_embeddings = np.load("models/ada_embeddings/holdout_embeddings.npy")      # (5000, 1536)
base_st_embeddings = np.load("models/sentence_transformer/holdout_embeddings.npy")  # (5000, 384)
finetuned_embeddings = np.load("models/finetuned_st/holdout_embeddings.npy")  # (5000, 384)
```

### Step 2: Compute Similarity Matrices

```python
# Compute all pairwise cosine similarities
ada_sims = cosine_similarity(ada_embeddings)           # (5000, 5000)
base_sims = cosine_similarity(base_st_embeddings)      # (5000, 5000)
finetuned_sims = cosine_similarity(finetuned_embeddings)  # (5000, 5000)
```

Each `sims[i][j]` = cosine similarity between paper i and paper j.

### Step 3: Sample Query Papers

```python
# Randomly sample n query papers
np.random.seed(42)
query_indices = np.random.choice(n_papers, n_samples, replace=False)
```

### Step 4: For Each Query, Compare Rankings

```python
for query_idx in query_indices:
    # Ground truth: Ada's top-k
    ada_top_k = get_top_k_indices(ada_sims, query_idx, k)
    
    # Model's ranking
    model_top_k = get_top_k_indices(model_sims, query_idx, k)
    
    # Calculate overlap and ranking quality metrics
    metrics = calculate_metrics(ada_top_k, model_top_k, k)
```

### Step 5: Aggregate Results

```python
# Average metrics across all query samples
mean_recall = np.mean(all_recall_scores)
mean_ndcg = np.mean(all_ndcg_scores)
# etc.
```

---

## Metrics Explained

### 1. Recall@k

**Question**: What fraction of Ada's top-k recommendations did the model find?

```python
def recall_at_k(ada_top_k, model_top_k, k):
    ada_set = set(ada_top_k[:k])
    model_set = set(model_top_k[:k])
    overlap = len(ada_set & model_set)
    return overlap / k
```

**Example**:
```
Ada's top-5:    [7, 23, 156, 89, 42]
Model's top-5:  [7, 89, 12, 23, 99]

Overlap: {7, 23, 89} = 3 papers
Recall@5 = 3/5 = 0.60
```

**Interpretation**: The model found 60% of Ada's top recommendations.

---

### 2. NDCG@k (Normalized Discounted Cumulative Gain)

**Question**: How well does the model rank relevant items? (Position matters!)

NDCG rewards:
- Finding relevant items (in Ada's top-k)
- Ranking them higher in the list

```python
def ndcg_at_k(ada_top_k, model_ranking, k):
    ada_set = set(ada_top_k[:k])
    
    # Relevance: 1 if in Ada's top-k, else 0
    relevances = [1.0 if idx in ada_set else 0.0 for idx in model_ranking[:k]]
    
    # DCG: sum of relevance / log2(position + 1)
    # Position 1 → divide by log2(2) = 1.0
    # Position 2 → divide by log2(3) = 1.58
    # Position 3 → divide by log2(4) = 2.0
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    
    # Ideal DCG: if we ranked all relevant items first
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ada_set))))
    
    return dcg / idcg if idcg > 0 else 0.0
```

**Example**:
```
Ada's top-3: [7, 23, 156]
Model ranking: [7, 99, 23, 156, 12]  (positions 1, 3, 4)

Relevances: [1, 0, 1, 1, 0] (at positions 1-5)

DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) + 1/log2(5) + 0/log2(6)
    = 1.0 + 0 + 0.5 + 0.43 + 0
    = 1.93

IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) (ideal: all 3 at top)
     = 1.0 + 0.63 + 0.5
     = 2.13

NDCG@5 = 1.93 / 2.13 = 0.91
```

**Interpretation**: Higher NDCG means relevant items appear earlier in the ranking.

---

### 3. MRR (Mean Reciprocal Rank)

**Question**: How quickly does the model find the first relevant item?

```python
def mrr(ada_top_k, model_ranking, k):
    ada_set = set(ada_top_k[:k])
    
    for position, idx in enumerate(model_ranking[:k]):
        if idx in ada_set:
            return 1.0 / (position + 1)  # 1-indexed position
    
    return 0.0  # No relevant item found
```

**Example**:
```
Ada's top-5: [7, 23, 156, 89, 42]
Model ranking: [99, 7, 23, ...]

First relevant item (7) is at position 2
MRR = 1/2 = 0.50
```

**Interpretation**: 
- MRR = 1.0 → first item is relevant
- MRR = 0.5 → first relevant item at position 2
- MRR = 0.1 → first relevant item at position 10

---

### 4. MAP@k (Mean Average Precision)

**Question**: On average, how precise is the model at each point where it finds a relevant item?

```python
def map_at_k(ada_top_k, model_ranking, k):
    ada_set = set(ada_top_k[:k])
    
    precisions = []
    relevant_count = 0
    
    for position, idx in enumerate(model_ranking[:k]):
        if idx in ada_set:
            relevant_count += 1
            precision_at_this_point = relevant_count / (position + 1)
            precisions.append(precision_at_this_point)
    
    return np.mean(precisions) if precisions else 0.0
```

**Example**:
```
Ada's top-5: [7, 23, 156, 89, 42]
Model ranking: [7, 99, 23, 156, 12]

Position 1: 7 is relevant → precision = 1/1 = 1.0
Position 2: 99 not relevant → skip
Position 3: 23 is relevant → precision = 2/3 = 0.67
Position 4: 156 is relevant → precision = 3/4 = 0.75
Position 5: 12 not relevant → skip

MAP@5 = mean(1.0, 0.67, 0.75) = 0.81
```

**Interpretation**: MAP penalizes relevant items appearing later more than NDCG.

---

### 5. Spearman Correlation (Training Metric)

**Question**: How well does the model's similarity ordering correlate with Ada's?

Used during training to evaluate the fine-tuned model:

```python
from scipy.stats import spearmanr

# Sample pairwise similarities
ada_sims = [0.85, 0.72, 0.91, 0.68, ...]
model_sims = [0.79, 0.65, 0.88, 0.70, ...]

spearman_correlation, p_value = spearmanr(ada_sims, model_sims)
# Returns: ~0.85 (good correlation)
```

**Interpretation**:
- 1.0 = perfect agreement in ranking
- 0.0 = no correlation
- -1.0 = perfect inverse ranking

---

## Metrics Summary Table

| Metric | Range | What it Measures | Higher is Better? |
|--------|-------|------------------|-------------------|
| Recall@k | 0-1 | Coverage of relevant items | ✓ |
| NDCG@k | 0-1 | Ranking quality (position-aware) | ✓ |
| MRR | 0-1 | Speed to first relevant item | ✓ |
| MAP@k | 0-1 | Average precision at relevant points | ✓ |
| Spearman ρ | -1 to 1 | Rank correlation with Ada | ✓ |
| MAE | 0+ | Mean absolute error from Ada | ✗ |

---

## Training vs Evaluation Metrics

### During Training (Validation Set)

The `EmbeddingSimilarityEvaluator` computes **Spearman correlation** between:
- Model's predicted similarities for validation pairs
- Target similarities (ada_sim values)

```python
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sentences1,
    sentences2=val_sentences2,
    scores=val_ada_sims,  # Target scores
    name='arxiv-validation'
)
# Reports: eval_arxiv-validation_spearman_cosine
```

This is computed on the **training pairs** (not full corpus).

### During Final Evaluation (Holdout Set)

The `evaluate_vs_ada.py` script computes **ranking metrics** (Recall, NDCG, MRR, MAP) by:
- Using full similarity matrices over 5,000 holdout papers
- Comparing top-k rankings between models and Ada

This tests **retrieval performance** on unseen papers.

---

## Running Evaluations

### Quick Evaluation

```bash
# Evaluate with 500 samples at k=1,3,5,10
python -m scripts.evaluate_vs_ada --n-samples 500 --k-values 1 3 5 10
```

### Full Pipeline

```bash
# 1. Generate holdout embeddings (if not done)
make embeddings

# 2. Run evaluation
make eval
```

### Output

Results are saved to `results/ada_eval/`:
- `metrics_vs_k.png` - Visualization of all metrics
- `{model}_results.csv` - Detailed per-k metrics

### Example Output

```
EVALUATION RESULTS (vs Ada-002 as Ground Truth)
======================================================================

base_st:
  k      Recall          NDCG            MRR             MAP            
  ------ --------------- --------------- --------------- ---------------
  1      0.156±0.363    0.156±0.363    0.156±0.363    0.156±0.363
  3      0.187±0.224    0.198±0.227    0.234±0.352    0.189±0.226
  5      0.198±0.186    0.213±0.192    0.267±0.347    0.195±0.181
  10     0.215±0.138    0.234±0.149    0.298±0.339    0.208±0.137

finetuned_st:
  k      Recall          NDCG            MRR             MAP            
  ------ --------------- --------------- --------------- ---------------
  1      0.234±0.424    0.234±0.424    0.234±0.424    0.234±0.424
  3      0.267±0.251    0.285±0.258    0.334±0.382    0.271±0.255
  5      0.289±0.198    0.312±0.207    0.367±0.371    0.293±0.195
  10     0.312±0.152    0.341±0.162    0.398±0.358    0.315±0.149
```

**Interpretation**: Fine-tuned model shows ~50% relative improvement over base ST across all metrics.

---

## Visual Interpretation

```
                    Recall@k vs k
    1.0 ┤
        │                              ╭── finetuned_st
    0.8 ┤                         ╭────╯
        │                    ╭────╯
    0.6 ┤               ╭────╯
        │          ╭────╯
    0.4 ┤     ╭────╯         ╭── base_st
        │╭────╯         ╭────╯
    0.2 ┤          ╭────╯
        │     ╭────╯
    0.0 ┼─────┴────┴────┴────┴────┴───
        1    3    5    10   20   50  k

Higher curve = better model
```

---

## Key Takeaways

1. **Recall@k** tells you coverage - "did you find the relevant items?"
2. **NDCG@k** tells you ranking quality - "did you rank them well?"
3. **MRR** tells you responsiveness - "how fast to first good result?"
4. **MAP@k** balances precision across the ranking
5. **Spearman correlation** (training) measures similarity score alignment

For a recommendation system, **NDCG and Recall** are typically most important - you want to find relevant items AND rank them at the top.
