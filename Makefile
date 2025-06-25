.PHONY: install dev-install test clean train

install:
	pip install -e .

dev-install:
	pip install -e ".[dev,notebooks]"

test:
	python -m pytest tests/ -v

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

train:
	python scripts/train_model.py --epochs 3 --batch-size 4

generate:
	python scripts/run_generator.py --premise "∀x (P(x) → Q(x)) ∧ ∃x P(x)"

setup-dirs:
	mkdir -p data models/trained_model logs tests

format:
	black src/ scripts/ tests/

lint:
	flake8 src/ scripts/ tests/