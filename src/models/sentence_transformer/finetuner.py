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


def load_training_data(annotations_path):
    """
    Load pre-balanced training data from mine_hard_pairs.py.
    
    The mining script already creates balanced positive/negative pairs
    using percentile-based thresholds. We just load and use them directly.
    
    Args:
        annotations_path: Path to hard pairs CSV (already balanced)
    
    Returns:
        HuggingFace Dataset with sentence1, sentence2, and score columns
    """
    df = pd.read_csv(annotations_path)
    logger.info(f"Loaded {len(df)} training pairs")
    
    if 'ada_sim' not in df.columns:
        raise ValueError("Training data must have 'ada_sim' column")
    
    # Use ada_sim as the target score - mining script already balanced the data
    data = {
        'sentence1': df['text1'].tolist(),
        'sentence2': df['text2'].tolist(),
        'score': df['ada_sim'].tolist(),
    }
    
    scores = data['score']
    logger.info(f"Score distribution: min={min(scores):.3f}, max={max(scores):.3f}, mean={sum(scores)/len(scores):.3f}")
    
    return Dataset.from_dict(data)


def finetune_sentence_transformer(
    papers_df: pd.DataFrame = None,
    annotations_path=None,
    base_model: str = None,
    epochs: int = None,
    batch_size: int = None,
    warmup_ratio: float = 0.1,
    evaluation_steps: int = None,
    output_path=None,
    test_size: float = 0.2,
    plot_training_curve: bool = True,
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
    logger.info(f"  Annotations path: {annotations_path}")
    
    # Load and prepare data as HuggingFace Dataset
    dataset = load_training_data(annotations_path)
    
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
    
    # Define loss function - CosineSimilarityLoss with balanced pos/neg pairs
    # Positive pairs (high ada_sim) push embeddings together
    # Negative pairs (low ada_sim) push embeddings apart - prevents collapse!
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
    train_result = trainer.train()
    
    # Save the final model
    model.save(str(output_path))
    logger.info(f"Fine-tuning complete! Model saved to: {output_path}")
    
    # Plot training curve if requested
    if plot_training_curve:
        plot_training_loss(trainer, output_path)
    
    return model


def plot_training_loss(trainer, output_path):
    """
    Plot and save training loss curve.
    
    Args:
        trainer: SentenceTransformerTrainer with training history
        output_path: Directory to save plot
    """
    import matplotlib.pyplot as plt
    
    try:
        # Extract training history from trainer state
        history = trainer.state.log_history
        
        # Separate training and eval metrics
        train_steps = []
        train_losses = []
        eval_steps = []
        eval_scores = []
        
        for entry in history:
            if 'loss' in entry:
                train_steps.append(entry.get('step', 0))
                train_losses.append(entry['loss'])
            if 'eval_arxiv-validation_spearman_cosine' in entry:
                eval_steps.append(entry.get('step', 0))
                eval_scores.append(entry['eval_arxiv-validation_spearman_cosine'])
        
        if not train_losses:
            logger.warning("No training history found for plotting")
            return
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot training loss
        axes[0].plot(train_steps, train_losses, 'b-', linewidth=2, label='Training Loss')
        axes[0].set_xlabel('Step', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot validation score
        if eval_scores:
            axes[1].plot(eval_steps, eval_scores, 'g-o', linewidth=2, markersize=6, label='Spearman Correlation')
            axes[1].set_xlabel('Step', fontsize=12)
            axes[1].set_ylabel('Spearman Correlation', fontsize=12)
            axes[1].set_title('Validation Performance', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No validation scores available', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(output_path).parent / "training_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curve saved to {plot_path}")
        
    except Exception as e:
        logger.warning(f"Could not plot training curve: {e}")


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
