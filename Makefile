# Makefile
.PHONY: data annotate train finetune eval all api-dev api-build api-up api-down deploy

# ============== Training Pipeline ==============
data:
	python -m scripts.prepare_data

annotate:
	python -m scripts.generate_annotations

train:
	python -m scripts.train_models --baseline

finetune:
	python -m scripts.train_models --finetune

eval:
	python -m scripts.run_evaluation --n_samples 20 --top_k 10

all: data train annotate finetune eval

# ============== API Development ==============
api-dev:
	@echo "Starting API services for local development..."
	@echo "TF-IDF: http://localhost:8001"
	@echo "ST: http://localhost:8002"
	@echo "Gateway: http://localhost:8000"
	python -m api.services.tfidf.main & \
	python -m api.services.sentence_transformer.main & \
	python -m api.gateway.main

api-tfidf:
	python -m api.services.tfidf.main

api-st:
	python -m api.services.sentence_transformer.main

api-gateway:
	python -m api.gateway.main

# ============== Docker ==============
api-build:
	docker-compose build

api-up:
	docker-compose up -d

api-down:
	docker-compose down

# ============== Deployment ==============
deploy:
	chmod +x deploy.sh && ./deploy.sh
