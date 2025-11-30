"""
Generate LLM-annotated training pairs.
"""
import asyncio
import csv
import logging
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

from src.config import settings
from src.logging_config import setup_logging
from src.annotation.pair_sampler import sample_diverse_pairs
from src.annotation.llm_scorer import DualLLMScorer

logger = logging.getLogger(__name__)


async def main():
    setup_logging()
    
    # Load processed dataset
    df = pd.read_parquet(settings.data.dataset.paths.processed_data)
    logger.info(f"Loaded {len(df)} papers")
    
    # Set 'id' as index for fast lookup
    df = df.set_index('id')
    
    # Sample diverse pairs (uses original df with 'id' column, so pass reset version)
    n_pairs = settings.data.annotation.n_pairs
    pairs = sample_diverse_pairs(df.reset_index(), n_pairs)

    logger.info(f"Sampled {len(pairs)} pairs")
    
    # Initialize scorer
    scorer = DualLLMScorer()
    
    # Score all pairs
    results = []
    batch_size = settings.data.annotation.batch_size
    output_path = Path(settings.data.annotation.paths.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for existing results to resume
    if output_path.exists():
        # Read with explicit string dtype for paper IDs to preserve leading zeros
        existing_df = pd.read_csv(output_path, dtype={'paper1_id': str, 'paper2_id': str})
        existing_pairs = set(zip(existing_df['paper1_id'], existing_df['paper2_id']))
        pairs = [(p1, p2) for p1, p2 in pairs if (p1, p2) not in existing_pairs]
        results = existing_df.to_dict('records')
        logger.info(f"Resuming from {len(existing_pairs)} existing pairs, {len(pairs)} remaining")
    
    save_interval = 50  # Save every 50 batches
    
    for i in tqdm(range(0, len(pairs), batch_size), desc="Scoring pairs"):
        batch = pairs[i:i + batch_size]
        tasks = []
        
        for p1_id, p2_id in batch:
            query_paper = df.loc[p1_id].to_dict()
            recommendation = df.loc[p2_id].to_dict()
            tasks.append(scorer.score_pair(query_paper, recommendation))
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (p1_id, p2_id), score_data in zip(batch, batch_results):
            if isinstance(score_data, Exception):
                logger.warning(f"Failed to score pair ({p1_id}, {p2_id}): {score_data}")
                continue
            if score_data:
                results.append({
                    'paper1_id': str(p1_id),  # Ensure string to preserve leading zeros
                    'paper2_id': str(p2_id),
                    **score_data
                })
        
        # Save incrementally every N batches
        batch_num = i // batch_size
        if batch_num > 0 and batch_num % save_interval == 0:
            pd.DataFrame(results).to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
            logger.info(f"Checkpoint saved: {len(results)} pairs")
        
        # Small delay between batches to respect rate limits
        await asyncio.sleep(0.5)
    
    # Final save
    results_df = pd.DataFrame(results)
    # Use QUOTE_ALL and escapechar to handle special characters in reasoning fields
    results_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    logger.info(f"Saved {len(results_df)} annotated pairs to {output_path}")
    
    # Log distribution
    logger.info("Score distribution:")
    logger.info(f"\n{results_df['avg_score'].describe()}")


if __name__ == "__main__":
    asyncio.run(main())