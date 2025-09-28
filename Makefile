.PHONY: help build run test fmt lint bench clean install dev-setup data index api

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install package in editable mode"
	@echo "  make dev-setup    - Setup development environment"
	@echo "  make build        - Build Docker image"
	@echo "  make run          - Start services with docker-compose"
	@echo "  make test         - Run tests with coverage"
	@echo "  make fmt          - Format code"
	@echo "  make lint         - Run linting and type checking"
	@echo "  make bench        - Run benchmarks and evaluation"
	@echo "  make data         - Download DCASE dataset"
	@echo "  make index        - Index audio files into Qdrant"
	@echo "  make api          - Start API server"
	@echo "  make clean        - Clean up containers and volumes"

# Installation and setup
install:
	pip install -e .[dev]

dev-setup:
	conda env create -f env.yml --force
	conda run -n ml_py310 pip install -e .[dev]

# Docker operations
build:
	docker build -t industrial-audio-rag .

run:
	docker-compose up -d

clean:
	docker-compose down -v
	docker system prune -f

# Code quality
test:
	pytest tests/ -v --cov=src/rag_audio --cov-report=term-missing --cov-report=html

fmt:
	ruff format .

lint:
	ruff check .
	mypy src/rag_audio/

# Benchmarking and evaluation
bench:
	@if [ -f eval/benchmark.py ]; then python eval/benchmark.py; else echo "Benchmark script not implemented yet"; fi

# Data operations
data:
	bash scripts/get_dcase24.sh

index:
	python -m rag_audio.indexer --data Data/Dcase --collection dcase24_bearing

# Development server
api:
	uvicorn rag_audio.api:app --reload --host 0.0.0.0 --port 8000

# Quality gate (for CI)
quality-gate: lint test bench
	@echo "✅ All quality checks passed"