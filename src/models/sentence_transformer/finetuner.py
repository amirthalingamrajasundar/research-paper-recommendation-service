"""
Fine-tune Sentence Transformer on annotated paper pairs.
"""
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import BatchSamplers

from src.config import settings

logger = logging.getLogger(__name__)


def load_training_data(annotations_path, papers_df):
    """
    Load annotations and convert to HuggingFace Dataset format.
    
    Args:
        annotations_path: Path to annotated pairs CSV
        papers_df: DataFrame with all papers (must have 'id', 'text' columns)
    
    Returns:
        HuggingFace Dataset with sentence1, sentence2, and score columns
    """
    # Load annotations with explicit string dtype to preserve paper ID format
    annotations_df = pd.read_csv(annotations_path, dtype={'paper1_id': str, 'paper2_id': str})
    logger.info(f"Loaded {len(annotations_df)} annotated pairs")
    
    # Index papers by ID for fast lookup
    papers_df = papers_df.set_index('id')
    
    # Build training data
    data = {'sentence1': [], 'sentence2': [], 'score': []}
    skipped = 0
    
    for _, row in annotations_df.iterrows():
        try:
            paper1 = papers_df.loc[row['paper1_id']]
            paper2 = papers_df.loc[row['paper2_id']]
            
            data['sentence1'].append(paper1['text'])
            data['sentence2'].append(paper2['text'])
            data['score'].append(float(row['avg_score']))  # normalized 0-1
        except KeyError:
            skipped += 1
            continue
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} pairs (paper IDs not found)")
    
    logger.info(f"Created {len(data['sentence1'])} training examples")
    return Dataset.from_dict(data)


def finetune_sentence_transformer(
    papers_df: pd.DataFrame,
    annotations_path=None,
    base_model: str = None,
    epochs: int = None,
    batch_size: int = None,
    warmup_ratio: float = 0.1,
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
        warmup_ratio: Fraction of steps for warmup
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
    evaluation_steps = evaluation_steps or ft_config.evaluation_steps
    output_path = Path(output_path or ft_config.paths.model)
    
    logger.info("Fine-tuning configuration:")
    logger.info(f"  Base model: {base_model}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Output path: {output_path}")
    
    # Load and prepare data as HuggingFace Dataset
    dataset = load_training_data(annotations_path, papers_df)
    
    if len(dataset) < 10:
        raise ValueError(f"Not enough training examples ({len(dataset)}). Need at least 10.")
    
    # Split into train/validation
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset)}")
    
    # Load base model
    logger.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    
    # Define loss function - CosineSimilarityLoss for regression on similarity scores
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    # Define evaluator for validation
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset['sentence1'],
        sentences2=eval_dataset['sentence2'],
        scores=eval_dataset['score'],
        name='arxiv-validation'
    )
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        eval_strategy="steps",
        eval_steps=evaluation_steps,
        save_strategy="steps",
        save_steps=evaluation_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_arxiv-validation_spearman_cosine",
        greater_is_better=True,
        logging_steps=50,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    
    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    
    # Fine-tune
    logger.info(f"Starting fine-tuning for {epochs} epochs...")
    trainer.train()
    
    # Save the final model
    model.save(str(output_path))
    logger.info(f"Fine-tuning complete! Model saved to: {output_path}")
    
    return model


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
