# scripts/debug_embeddings.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
base = np.load("models/sentence_transformer/holdout_embeddings.npy")
finetuned = np.load("models/finetuned_st/holdout_embeddings.npy")
ada = np.load("models/ada_embeddings/holdout_embeddings.npy")

# Check if embeddings are different
print(f"Base vs Finetuned embeddings identical: {np.allclose(base, finetuned)}")
print(f"Base shape: {base.shape}, Finetuned shape: {finetuned.shape}")

# Check similarity score distributions for paper 0
idx = 0
base_sims = cosine_similarity([base[idx]], base)[0]
ft_sims = cosine_similarity([finetuned[idx]], finetuned)[0]
ada_sims = cosine_similarity([ada[idx]], ada)[0]

print(f"\nSimilarity distributions for paper 0:")
print(f"  Base ST:    min={base_sims.min():.3f}, max={base_sims.max():.3f}, mean={base_sims.mean():.3f}")
print(f"  Finetuned:  min={ft_sims.min():.3f}, max={ft_sims.max():.3f}, mean={ft_sims.mean():.3f}")
print(f"  Ada:        min={ada_sims.min():.3f}, max={ada_sims.max():.3f}, mean={ada_sims.mean():.3f}")

# Check top-10 overlap
base_top10 = np.argsort(base_sims)[::-1][1:11]
ft_top10 = np.argsort(ft_sims)[::-1][1:11]
ada_top10 = np.argsort(ada_sims)[::-1][1:11]

print(f"\nTop-10 overlap with Ada:")
print(f"  Base ST:   {len(set(ada_top10) & set(base_top10))}/10")
print(f"  Finetuned: {len(set(ada_top10) & set(ft_top10))}/10")
print(f"\nAre rankings different? {not np.array_equal(base_top10, ft_top10)}")