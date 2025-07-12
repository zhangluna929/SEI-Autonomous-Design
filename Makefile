.PHONY: install clean test data pretrain finetune generate demo

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

# Data preparation
data:
	python build_master_dataset.py --out master.parquet

# Model training pipeline
pretrain:
	python pretrain/train_lightning.py

finetune: pretrain
	python finetune/train_predictor.py

generate: finetune
	python generator/run_generation.py

# Multi-scale simulation
multiscale:
	python multiscale/dual_active_learning.py
	python multiscale/phase_field.py

# Reinforcement learning
rl:
	python rl_generation/chemgpt_rl.py

# Run comprehensive demo
demo:
	python demo_multiscale_rl.py

# Testing
test:
	python -m pytest tests/ -v

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ temp/ tmp/
	rm -rf test_*/ demo_*/
	rm -f *.log *.tmp

# Full pipeline
all: install data pretrain finetune generate multiscale rl

# Help
help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  data       - Prepare datasets"
	@echo "  pretrain   - Train foundation models"
	@echo "  finetune   - Fine-tune predictors"
	@echo "  generate   - Train generators"
	@echo "  multiscale - Run multi-scale simulations"
	@echo "  rl         - Run reinforcement learning"
	@echo "  demo       - Run comprehensive demo"
	@echo "  test       - Run tests"
	@echo "  clean      - Clean temporary files"
	@echo "  all        - Run full pipeline" 