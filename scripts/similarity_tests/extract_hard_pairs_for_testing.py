# scripts/extract_hard_pairs_for_validation.py
"""Extract a sample of hard pairs for manual validation."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings

# Load hard pairs
hard_pairs_path = settings.data.hard_pairs.output
df = pd.read_csv(hard_pairs_path)
print(f"Loaded {len(df)} pairs from {hard_pairs_path}")

# Categories to validate
samples = []

# 1. Hardest positives (highest gap: ada high, base low)
hardest_pos = df[df['ada_sim'] > 0.8].nlargest(10, 'gap')
hardest_pos['category'] = 'hard_positive'
samples.append(hardest_pos)

# 2. Hardest negatives (low ada, high base - base is wrong)
hardest_neg = df[df['ada_sim'] < 0.7].nsmallest(10, 'gap')  # Most negative gap
hardest_neg['category'] = 'hard_negative'
samples.append(hardest_neg)

# 3. Borderline cases (ada_sim near decision boundary ~0.7-0.75)
borderline = df[(df['ada_sim'] >= 0.70) & (df['ada_sim'] <= 0.75)].sample(min(10, len(df[(df['ada_sim'] >= 0.70) & (df['ada_sim'] <= 0.75)])), random_state=42)
borderline['category'] = 'borderline'
samples.append(borderline)

# 4. High agreement (sanity check - both models agree)
agreement = df[(abs(df['gap']) < 0.1) & (df['ada_sim'] > 0.8)].sample(min(5, len(df[(abs(df['gap']) < 0.1) & (df['ada_sim'] > 0.8)])), random_state=42)
agreement['category'] = 'agreement'
samples.append(agreement)

# 5. True negatives (lowest ada_sim - genuinely dissimilar pairs)
true_neg = df.nsmallest(10, 'ada_sim').copy()
true_neg['category'] = 'true_negative'
samples.append(true_neg)

# Combine and format for review
validation_df = pd.concat(samples, ignore_index=True)

# Format output for easier reading
output = []
for _, row in validation_df.iterrows():
    output.append({
        'category': row['category'],
        'ada_sim': round(row['ada_sim'], 3),
        'base_sim': round(row['base_sim'], 3),
        'gap': round(row['gap'], 3),
        'text1':row['text1'],
        'text2': row['text2'],
        'id1': row['id1'],
        'id2': row['id2'],
    })

output_df = pd.DataFrame(output)

# Save for review
output_path = PROJECT_ROOT / 'data/annotated/validation_sample.csv'
output_df.to_csv(output_path, index=False)
print(f"\nSaved {len(output_df)} pairs to {output_path}")

# Print summary
print(f"\n{'='*60}")
print("VALIDATION SAMPLE SUMMARY")
print(f"{'='*60}")
for cat in ['hard_positive', 'hard_negative', 'borderline', 'agreement', 'true_negative']:
    cat_df = output_df[output_df['category'] == cat]
    print(f"\n{cat.upper()} ({len(cat_df)} pairs):")
    if len(cat_df) > 0:
        print(f"  Ada sim: {cat_df['ada_sim'].mean():.3f} avg")
        print(f"  Base sim: {cat_df['base_sim'].mean():.3f} avg")
        print(f"  Gap: {cat_df['gap'].mean():.3f} avg")

# Preview one from each category
print(f"\n{'='*60}")
print("SAMPLE PAIRS FOR REVIEW (1 per category)")
print(f"{'='*60}")
for cat in ['hard_positive', 'hard_negative', 'borderline', 'agreement', 'true_negative']:
    cat_df = output_df[output_df['category'] == cat]
    if len(cat_df) > 0:
        row = cat_df.iloc[0]
        print(f"\n[{row['category']}] Ada: {row['ada_sim']}, Base: {row['base_sim']}, Gap: {row['gap']}")
        print(f"  TEXT 1: {row['text1']}")
        print(f"  TEXT 2: {row['text2']}")