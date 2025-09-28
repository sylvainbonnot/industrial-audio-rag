# Contributing to Industrial Audio RAG

Thank you for your interest in contributing to this project! This guide will help you get started.

## Development Setup

1. **Clone and setup environment:**
   ```bash
   git clone https://github.com/sylvainbonnot/industrial-audio-rag.git
   cd industrial-audio-rag
   conda env create -f env.yml && conda activate ml_py310
   pip install -e .[dev]
   ```

2. **Start services:**
   ```bash
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:v1.8.1
   ```

3. **Run tests:**
   ```bash
   make test
   ```

## Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and ensure quality:**
   ```bash
   make fmt    # Format code
   make lint   # Check linting
   make test   # Run tests
   ```

3. **Commit with clear messages:**
   ```bash
   git commit -m "feat: add new audio feature extraction method"
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Standards

- Use type hints for all functions
- Follow existing code style (ruff + mypy configured)
- Add tests for new functionality
- Update documentation as needed
- Keep commits atomic and well-described

## Pull Request Process

1. Ensure all tests pass and code quality checks succeed
2. Update README.md if you've changed functionality
3. Add tests that cover your changes
4. Request review from maintainers
5. Address feedback promptly

## Reporting Issues

- Use GitHub Issues to report bugs or request features
- Include reproduction steps for bugs
- Add relevant labels and assign to appropriate milestone

## Questions?

Feel free to open an issue for questions or reach out to maintainers.