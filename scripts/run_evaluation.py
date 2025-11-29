"""
Script to evaluate recommendation models.
"""
import argparse
import pickle
import pandas as pd

from src.config import settings
from src.models.tfidf.trainer import TFIDFModel
from src.models.sentence_transformer.base_model import SentenceTransformerModel
from src.evaluation.evaluator import (
    evaluate_models, 
    print_evaluation_summary, 
    compare_models
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation models")
    parser.add_argument('--n_samples', type=int, default=20, 
                        help='Number of query papers to sample')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of recommendations per query')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results pickle')
    args = parser.parse_args()
    
    # Load data
    print("Loading processed data...")
    df = pd.read_parquet(settings.data.dataset.paths.processed_data)
    print(f"Loaded {len(df)} papers")
    
    # Load models
    print("\nLoading models...")
    models = {}
    
    # TF-IDF
    try:
        tfidf_model = TFIDFModel()
        tfidf_model.load(df)
        models['tfidf'] = tfidf_model
        print("  ✓ TF-IDF loaded")
    except FileNotFoundError:
        print("  ✗ TF-IDF not found (run `make train` first)")
    
    # Sentence Transformer
    try:
        st_model = SentenceTransformerModel()
        st_model.load(df)
        models['sentence_transformer'] = st_model
        print("  ✓ Sentence Transformer loaded")
    except FileNotFoundError:
        print("  ✗ Sentence Transformer not found (run `make train` first)")
    
    # Fine-tuned Sentence Transformer
    try:
        ft_embeddings_path = settings.model.sentence_transformer.fine_tuning.paths.embeddings
        ft_model_path = settings.model.sentence_transformer.fine_tuning.paths.model
        
        ft_model = SentenceTransformerModel(model_name=str(ft_model_path))
        ft_model.load(df, embeddings_path=ft_embeddings_path)
        models['finetuned_st'] = ft_model
        print("  ✓ Fine-tuned Sentence Transformer loaded")
    except FileNotFoundError:
        print("  ✗ Fine-tuned model not found (run `make finetune` first)")
    
    if not models:
        print("\nNo models available for evaluation. Run training first.")
        return
    
    # Run evaluation
    print(f"\nRunning evaluation with {args.n_samples} samples, top_k={args.top_k}...")
    results = evaluate_models(
        models, df, 
        n_samples=args.n_samples, 
        top_k=args.top_k,
        random_seed=args.seed
    )
    
    # Print summary
    print_evaluation_summary(results)
    
    # Print comparison
    comparison = compare_models(results)
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(comparison.to_string())
    
    # Save results
    output_path = args.output or str(
        settings.model.tfidf.paths.vectorizer.parent.parent / "eval_results.pkl"
    )
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
