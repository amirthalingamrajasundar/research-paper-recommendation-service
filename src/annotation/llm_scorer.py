"""
Score paper pairs using multiple LLMs with structured output.
"""
import asyncio
import logging
import os
import textwrap

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


class ScoredPaper(BaseModel):
    """Structured output for paper relevance scoring."""
    score: int = Field(description="Relevance score from 1-5", ge=1, le=5)
    reasoning: str = Field(description="Brief explanation for the score")


class DualLLMScorer:
    # Max characters for abstract to avoid token limits
    MAX_ABSTRACT_LENGTH = 1000
    
    def __init__(self):
        # Initialize LLMs with structured output
        self.llms = {
            'gpt-4o': ChatOpenAI(
                model=os.getenv("GPT_4O"),
                temperature=0.1,
                openai_api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
                max_tokens=256,  # Limit response length
            ).with_structured_output(ScoredPaper),
            'claude-3.5-haiku': ChatOpenAI(
                model=os.getenv("CLAUDE_3_5_HAIKU"),
                temperature=0.1,
                openai_api_key=os.getenv("LLM_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL"),
                max_tokens=256,  # Limit response length
            ).with_structured_output(ScoredPaper),
        }
    
    def _truncate_text(self, text: str, max_length: int = None) -> str:
        """Truncate text to max length."""
        max_length = max_length or self.MAX_ABSTRACT_LENGTH
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def _build_prompt(self, query_paper, recommendation):
        """Build the evaluation prompt with truncated abstracts."""
        # Truncate abstracts to avoid token limit issues
        query_abstract = self._truncate_text(query_paper.get('abstract', ''))
        rec_abstract = self._truncate_text(recommendation.get('abstract', ''))
        
        return textwrap.dedent(f"""
            Rate the relevance of the recommended paper to the query paper (1-5 scale).
            
            QUERY PAPER:
            Title: {query_paper.get('title', '')}
            Abstract: {query_abstract}
            Categories: {query_paper.get('categories', '')}
            
            RECOMMENDED PAPER:
            Title: {recommendation.get('title', '')}
            Abstract: {rec_abstract}
            Categories: {recommendation.get('categories', '')}
            
            SCORING:
            5 = Highly relevant (same topic/methods)
            4 = Relevant (related field)
            3 = Moderately relevant (same broad area)
            2 = Weakly relevant (distant connection)
            1 = Not relevant
            
            Provide score and brief reasoning.
        """).strip()
    
    async def _score_with_llm(self, llm_name, llm, prompt, max_retries=3):
        """Score using a single LLM asynchronously with retry logic."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await llm.ainvoke(prompt)
                # Small delay after successful call to avoid rate limits
                await asyncio.sleep(0.5)
                return llm_name, response.score, response.reasoning
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Determine if we should retry
                should_retry = any([
                    "429" in str(e),
                    "rate" in error_str,
                    "validation" in error_str,
                    "json" in error_str,
                    "parsing" in error_str,
                    "eof" in error_str,
                    "missing" in error_str,
                ])
                
                if should_retry and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    logger.warning(f"{llm_name} error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{llm_name} error: {e}")
                    return llm_name, None, None
        
        logger.error(f"{llm_name} failed after {max_retries} retries: {last_error}")
        return llm_name, None, None
    
    async def score_pair(self, query_paper, recommendation):
        """Score a pair using all LLMs and average."""
        prompt = self._build_prompt(query_paper, recommendation)
        
        # Run all LLMs in parallel
        tasks = [
            self._score_with_llm(name, llm, prompt)
            for name, llm in self.llms.items()
        ]
        results = await asyncio.gather(*tasks)
        
        # Collect scores
        scores_dict = {}
        valid_scores = []
        for llm_name, score, reasoning in results:
            scores_dict[f'{llm_name}_score'] = score
            scores_dict[f'{llm_name}_reasoning'] = reasoning
            if score is not None:
                valid_scores.append(score)
        
        if valid_scores:
            # Average and normalize to 0-1 scale
            avg_raw = sum(valid_scores) / len(valid_scores)
            avg_normalized = (avg_raw - 1) / 4 
            
            return {
                **scores_dict,
                'avg_score_raw': avg_raw,
                'avg_score': avg_normalized
            }
        return None