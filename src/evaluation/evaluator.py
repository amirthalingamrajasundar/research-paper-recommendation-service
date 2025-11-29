"""
LLM-based evaluation for recommendation models.
"""
import os
import asyncio
import textwrap
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings

load_dotenv()


class ScoredPaper(BaseModel):
    """Structured output for paper relevance scoring."""
    score: int = Field(description="Relevance score (1-5)", ge=1, le=5)
    reasoning: str = Field(description="Reasoning for the score")


class LLMEvaluator:
    """Evaluator that uses LLMs to score recommendation quality."""
    
    def __init__(self):
        self.llms = {
            'gpt-4o': ChatOpenAI(
                model=os.getenv("GPT_4O"),
                temperature=0.1,
                openai_api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            ).with_structured_output(ScoredPaper),
            'claude-3.5-haiku': ChatOpenAI(
                model=os.getenv("CLAUDE_3_5_HAIKU"),
                temperature=0.1,
                openai_api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
            ).with_structured_output(ScoredPaper),
        }
    
    def _build_prompt(self, query_paper: Dict, recommendation: Dict) -> str:
        """Build the evaluation prompt."""
        return textwrap.dedent(f"""
            You are an expert research paper recommendation evaluator. Evaluate the relevance of the given recommended paper to the given query paper.
            
            QUERY PAPER:
                Title: {query_paper.get('title', '')}
                Abstract: {query_paper.get('abstract', '')}
                Categories: {query_paper.get('categories', '')}
            
            RECOMMENDATION TO EVALUATE:
                Title: {recommendation.get('title', '')}
                Abstract: {recommendation.get('abstract', '')}
                Categories: {recommendation.get('categories', '')}
                
            SCORING CRITERIA:
                5 = Highly relevant (same specific topic, methods, or direct extension)
                4 = Relevant (closely related field, similar techniques)  
                3 = Moderately relevant (same broad area, some conceptual overlap)
                2 = Weakly relevant (distant connection, different subfield)
                1 = Not relevant (unrelated topic)
            
            Consider:
                - Topic alignment and research questions
                - Methodological similarity
                - Shared keywords and concepts
                - Potential for citation
                - Same problem domain
        """).strip()
    
    def get_score_for_recommendation(self, llm, query_paper: Dict, recommendation: Dict) -> ScoredPaper:
        """Get score for a single recommendation using one LLM."""
        prompt = self._build_prompt(query_paper, recommendation)
        try:
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            print(f"LLM error: {e}")
            return ScoredPaper(score=3, reasoning="Error - defaulting to neutral score")
    
    def get_scores(self, query_paper: Dict, recommendations: List[Dict]) -> List[float]:
        """
        Get scores for all recommendations using LLM ensemble.
        
        Returns:
            List of average scores (1-5) for each recommendation
        """
        all_scores = defaultdict(dict)
        avg_scores = []
        
        for rec in recommendations:
            score_sum = 0
            for llm_name, llm in self.llms.items():
                response = self.get_score_for_recommendation(llm, query_paper, rec)
                score_sum += response.score
                all_scores[llm_name][rec.get('id', '')] = response
            
            avg_score = score_sum / len(self.llms)
            avg_scores.append(avg_score)
        
        return avg_scores


def calculate_metrics(scores: List[float], rec_categories: List[str], 
                      query_category: str, k: int = 10) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        scores: List of relevance scores (1-5) for each recommendation
        rec_categories: List of categories for each recommendation
        query_category: Category of the query paper
        k: Number of recommendations to consider
    
    Returns:
        Dictionary of metric names to values
    """
    scores = np.array(scores[:k])
    binary = (scores >= 3).astype(int)  # Relevant if score >= 3
    
    # Basic metrics
    precision = np.mean(binary)
    recall = np.sum(binary) / min(10, k)  # Assume max 10 relevant exist
    
    metrics = {
        'precision_at_k': precision,
        'recall_at_k': recall,
        'f1_at_k': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'avg_relevance': np.mean(scores),
        'category_consistency': np.mean([1 if cat == query_category else 0 for cat in rec_categories[:k]])
    }
    
    # MRR (Mean Reciprocal Rank)
    if np.any(binary):
        metrics['mrr'] = 1.0 / (np.argmax(binary) + 1)
    else:
        metrics['mrr'] = 0.0
    
    # MAP@K (Mean Average Precision)
    relevant_positions = np.where(binary)[0]
    if len(relevant_positions) > 0:
        precisions = [(i+1) / (pos+1) for i, pos in enumerate(relevant_positions)]
        metrics['map_at_k'] = np.mean(precisions)
    else:
        metrics['map_at_k'] = 0.0
    
    # nDCG@K (Normalized Discounted Cumulative Gain)
    dcg = np.sum((2**scores - 1) / np.log2(np.arange(1, k+1) + 1))
    ideal = np.sum((2**np.sort(scores)[::-1] - 1) / np.log2(np.arange(1, k+1) + 1))
    metrics['ndcg_at_k'] = dcg / ideal if ideal > 0 else 0.0
    
    return metrics


def evaluate_models(models: Dict[str, Any], df: pd.DataFrame, 
                    n_samples: int = 20, top_k: int = 10,
                    random_seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Evaluate multiple recommendation models using LLM-based scoring.
    
    Args:
        models: Dictionary of model_name -> model object
        df: DataFrame with all papers
        n_samples: Number of random query papers to sample
        top_k: Number of recommendations per query
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary of model_name -> DataFrame of metrics per sample
    """
    np.random.seed(random_seed)
    evaluator = LLMEvaluator()
    eval_indices = np.random.choice(len(df), n_samples, replace=False)
    
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        model_results = []
        
        for idx in tqdm(eval_indices, desc=f"Evaluating {model_name}"):
            # Get recommendations
            recs_df = model.get_recommendations_for_paper(idx, top_k=top_k)
            
            # Prepare data
            query = df.iloc[idx].to_dict()
            recs = [row['paper'] if isinstance(row['paper'], dict) else row['paper'].to_dict() 
                    for _, row in recs_df.iterrows()]
            rec_categories = [r.get('primary_category', '') for r in recs]
            
            # Get scores from LLMs
            scores = evaluator.get_scores(query, recs)
            
            # Calculate all metrics
            metrics = calculate_metrics(
                scores, rec_categories, 
                query.get('primary_category', ''), 
                k=top_k
            )
            model_results.append(metrics)
        
        all_results[model_name] = pd.DataFrame(model_results)
    
    return all_results


def print_evaluation_summary(results: Dict[str, pd.DataFrame]):
    """Print a summary of evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for model_name, results_df in results.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        for metric in results_df.columns:
            mean = results_df[metric].mean()
            std = results_df[metric].std()
            print(f"  {metric:20s}: {mean:.3f} ± {std:.3f}")


def compare_models(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a comparison DataFrame of all models.
    
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    comparison = []
    for model_name, results_df in results.items():
        row = {'model': model_name}
        for metric in results_df.columns:
            row[f'{metric}_mean'] = results_df[metric].mean()
            row[f'{metric}_std'] = results_df[metric].std()
        comparison.append(row)
    
    return pd.DataFrame(comparison).set_index('model')


if __name__ == "__main__":
    import pickle
    from src.models.tfidf.trainer import TFIDFModel
    from src.models.sentence_transformer.base_model import SentenceTransformerModel
    
    # Load data
    df = pd.read_parquet(settings.data.dataset.paths.processed_data)
    print(f"Loaded {len(df)} papers")
    
    # Load models
    print("\nLoading models...")
    tfidf_model = TFIDFModel()
    tfidf_model.load(df)
    
    st_model = SentenceTransformerModel()
    st_model.load(df)
    
    models = {
        'tfidf': tfidf_model,
        'sentence_transformer': st_model,
    }
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate_models(models, df, n_samples=5, top_k=5)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results
    output_path = settings.model.tfidf.paths.vectorizer.parent.parent / "eval_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")
