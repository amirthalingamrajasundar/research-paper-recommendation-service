# Makefile
.PHONY: data ada-embed mine-pairs train finetune embeddings eval all api-build api-up api-down deploy

# ============== Training Pipeline ==============
# Step 1: Prepare focused data with train/holdout split (settings from config)
data:
	python -m scripts.prepare_focused_data

# Step 2: Generate ada-002 embeddings (teacher model)
ada-embed:
	python -m scripts.generate_ada_embeddings

# Step 3: Mine hard training pairs (percentile-based: top 25% positive, bottom 25% negative)
mine-pairs:
	python -m scripts.mine_hard_pairs --top-k 20 --pos-percentile 75 --neg-percentile 25 --gap-threshold 0.2

# Step 4: Train baseline models (TF-IDF + base ST)
train:
	python -m scripts.train_models --baseline

# Step 5: Fine-tune on hard pairs
finetune:
	python -m scripts.train_models --finetune

# Step 6: Generate embeddings for holdout evaluation
embeddings:
	python -m scripts.generate_holdout_embeddings

# Step 7: Evaluate all models vs ada-002 (ground truth)
eval:
	python -m scripts.evaluate_vs_ada --n-samples 5000 --k-values 1 3 5 10

# Run complete pipeline
all: data ada-embed mine-pairs train finetune embeddings eval

# ============== API ==============
api-build:
	docker-compose build

api-up:
	docker-compose up -d

api-down:
	docker-compose down

# ============== Deployment ==============
deploy:
	chmod +x deploy.sh && ./deploy.sh
