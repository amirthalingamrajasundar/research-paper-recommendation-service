"""
Fine-tune Sentence Transformer on annotated paper pairs.
"""
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from src.config import settings

logger = logging.getLogger(__name__)


def load_training_data(annotations_path, papers_df):
    """
    Load annotations and convert to InputExample format.
    
    Args:
        annotations_path: Path to annotated pairs CSV
        papers_df: DataFrame with all papers (must have 'id', 'text' columns)
    
    Returns:
        List of InputExample objects
    """
    # Load annotations with explicit string dtype to preserve paper ID format
    annotations_df = pd.read_csv(annotations_path, dtype={'paper1_id': str, 'paper2_id': str})
    logger.info(f"Loaded {len(annotations_df)} annotated pairs")
    
    # Index papers by ID for fast lookup
    papers_df = papers_df.set_index('id')
    
    # Convert to InputExamples
    examples = []
    skipped = 0
    
    for _, row in annotations_df.iterrows():
        try:
            paper1 = papers_df.loc[row['paper1_id']]
            paper2 = papers_df.loc[row['paper2_id']]
            
            text1 = paper1['text']
            text2 = paper2['text']
            score = float(row['avg_score'])  # Already normalized 0-1
            
            examples.append(InputExample(
                texts=[text1, text2],
                label=score
            ))
        except KeyError:
            skipped += 1
            continue
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} pairs (paper IDs not found)")
    
    logger.info(f"Created {len(examples)} training examples")
    return examples


def finetune_sentence_transformer(
    papers_df: pd.DataFrame,
    annotations_path=None,
    base_model: str = None,
    epochs: int = None,
    batch_size: int = None,
    warmup_steps: int = None,
    evaluation_steps: int = None,
    output_path=None,
    test_size: float = 0.2
):
    """
    Fine-tune a sentence transformer model on annotated paper pairs.
    
    Args:
        papers_df: DataFrame with all papers
        annotations_path: Path to annotated pairs CSV
        base_model: Base model name (default from config)
        epochs: Number of training epochs
        batch_size: Training batch size
        warmup_steps: Warmup steps for scheduler
        evaluation_steps: Steps between evaluations
        output_path: Where to save the fine-tuned model
        test_size: Fraction of data for validation
    
    Returns:
        Fine-tuned SentenceTransformer model
    """
    # Load config defaults
    config = settings.model.sentence_transformer
    ft_config = config.fine_tuning
    
    annotations_path = annotations_path or settings.data.annotation.paths.output
    base_model = base_model or config.base_model
    epochs = epochs or ft_config.epochs
    batch_size = batch_size or ft_config.batch_size
    warmup_steps = warmup_steps or ft_config.warmup_steps
    evaluation_steps = evaluation_steps or ft_config.evaluation_steps
    output_path = output_path or ft_config.paths.model
    
    logger.info("Fine-tuning configuration:")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Output path: {output_path}")
    
    # Load and prepare data
    examples = load_training_data(annotations_path, papers_df)
    
    if len(examples) < 10:
        raise ValueError(f"Not enough training examples ({len(examples)}). Need at least 10.")
    
    # Split into train/validation
    train_examples, val_examples = train_test_split(
        examples, test_size=test_size, random_state=42
    )
    logger.info(f"Training samples: {len(train_examples)}")
    logger.info(f"Validation samples: {len(val_examples)}")
    
    # Load base model
    logger.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size
    )
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    # Define evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name='arxiv-validation'
    )
    
    # Calculate warmup steps if not specified
    if warmup_steps is None:
        warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Fine-tune!
    logger.info(f"Starting fine-tuning for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        save_best_model=True,
        show_progress_bar=True
    )
    
    logger.info(f"Fine-tuning complete! Model saved to: {output_path}")
    
    # Load the best model
    finetuned_model = SentenceTransformer(str(output_path))
    return finetuned_model


def generate_finetuned_embeddings(model, papers_df, output_path=None):
    """
    Generate embeddings using the fine-tuned model.
    
    Args:
        model: Fine-tuned SentenceTransformer
        papers_df: DataFrame with papers
        output_path: Where to save embeddings
    """
    import numpy as np
    
    output_path = output_path or settings.model.sentence_transformer.fine_tuning.paths.embeddings
    
    logger.info(f"Generating embeddings for {len(papers_df)} papers...")
    embeddings = model.encode(
        papers_df['text'].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    logger.info(f"Embeddings saved to {output_path}")
    
    return embeddings


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # Quick test
    papers_df = pd.read_parquet(settings.data.dataset.paths.processed_data)
    logger.info(f"Loaded {len(papers_df)} papers")
    
    # Fine-tune
    model = finetune_sentence_transformer(papers_df)
    
    # Generate embeddings
    generate_finetuned_embeddings(model, papers_df)
