# Evaluation Framework

Comprehensive evaluation and benchmarking framework for the Industrial Audio RAG system.

## 🎯 Overview

This evaluation framework provides automated quality assessment, performance benchmarking, and detailed reporting for the RAG system. It includes:

- **Quality Gates**: Automated pass/fail criteria for deployments
- **Performance Benchmarking**: Throughput, latency, and stress testing
- **Comprehensive Metrics**: 9 different evaluation dimensions
- **Visualization**: Charts and HTML reports
- **CI Integration**: Ready for automated pipelines

## 📁 Files

- `golden_prompts.jsonl` - Curated dataset of 20 test queries with expected outcomes
- `evaluation_schema.json` - JSON schema for response validation
- `quality_gate.py` - Main quality evaluation script with pass/fail criteria
- `benchmark.py` - Performance and stress testing framework
- `metrics.py` - Comprehensive evaluation metrics calculation
- `visualize.py` - Report generation with charts and HTML output

## 🚀 Quick Start

### 1. Run Quality Gate Evaluation

```bash
# Basic evaluation with default threshold (0.8)
python eval/quality_gate.py

# Custom threshold and output
python eval/quality_gate.py --threshold 0.85 --output eval/results/my_eval.json

# With API authentication
python eval/quality_gate.py --api-key your-api-key --threshold 0.9
```

### 2. Run Performance Benchmarks

```bash
# Performance benchmark
python eval/benchmark.py --suite performance --iterations 100

# Stress test
python eval/benchmark.py --suite stress --duration 300 --max-concurrent 50

# Complete benchmark suite
python eval/benchmark.py --suite all --output eval/results/full_benchmark.json
```

### 3. Generate Reports

```bash
# Generate quality evaluation report
python eval/visualize.py --input eval/results/evaluation_results.json --type quality

# Generate benchmark report with charts
python eval/visualize.py --input eval/results/benchmark_results.json --type benchmark
```

## 📊 Evaluation Metrics

### Quality Metrics (9 dimensions)

1. **Overall Score** (0-1): Weighted combination of all metrics
2. **Keyword Coverage** (0-1): Presence of expected keywords in answers
3. **Semantic Similarity** (0-1): Semantic alignment using embeddings
4. **Answer Quality** (0-1): Structure, coherence, and information density
5. **Retrieval Accuracy** (0-1): Accuracy of file/snippet retrieval
6. **Response Completeness** (0-1): How completely the query is addressed
7. **Factual Consistency** (0-1): Internal consistency and lack of contradictions
8. **Length Appropriateness** (0-1): Answer length vs expected range
9. **Technical Accuracy** (0-1): Domain-specific terminology and concepts

### Performance Metrics

- **Throughput**: Requests per second (RPS)
- **Response Time**: P50, P95, P99 percentiles
- **Error Rate**: Percentage of failed requests
- **System Utilization**: CPU, memory during tests
- **Stability Score**: Performance under stress

## 🔧 Configuration

### Quality Gate Thresholds

```python
# Default scoring weights in metrics.py
overall_score = (
    keyword_coverage * 0.15 +      # 15% - keyword presence
    semantic_similarity * 0.20 +   # 20% - semantic alignment  
    answer_quality * 0.25 +        # 25% - overall quality
    retrieval_accuracy * 0.10 +    # 10% - retrieval performance
    response_completeness * 0.15 + # 15% - completeness
    factual_consistency * 0.10 +   # 10% - consistency
    technical_accuracy * 0.05      # 5% - domain expertise
)
```

### Golden Dataset Format

```jsonl
{
  "query": "Which bearing clips in section 00 show dominant frequency above 900 Hz?",
  "expected_keywords": ["bearing", "section", "00", "900", "Hz", "frequency"],
  "expected_file_count": {"min": 1, "max": 10},
  "expected_answer_length": {"min": 50, "max": 500},
  "category": "frequency_analysis",
  "difficulty": "medium"
}
```

## 📈 CI/CD Integration

### GitHub Actions Integration

```yaml
# .github/workflows/quality-gate.yml
- name: Run Quality Gate
  run: |
    python eval/quality_gate.py --threshold 0.85 --output quality-results.json
    if [ $? -ne 0 ]; then
      echo "Quality gate failed!"
      exit 1
    fi

- name: Generate Report
  run: |
    python eval/visualize.py --input quality-results.json --type quality
    
- name: Upload Reports
  uses: actions/upload-artifact@v3
  with:
    name: evaluation-reports
    path: eval/reports/
```

### Make Integration

```bash
# Run evaluation via Makefile
make quality-gate    # Run quality evaluation
make benchmark       # Run performance tests  
make eval-report     # Generate comprehensive report
```

## 📋 Example Usage Patterns

### Development Workflow

```bash
# 1. Quick quality check during development
python eval/quality_gate.py --threshold 0.7

# 2. Full evaluation before PR
python eval/quality_gate.py --threshold 0.85 --output pre-pr-eval.json
python eval/visualize.py --input pre-pr-eval.json

# 3. Performance regression testing
python eval/benchmark.py --suite performance --iterations 50
```

### Production Deployment

```bash
# 1. Pre-deployment quality gate
python eval/quality_gate.py --threshold 0.9 --api-url https://staging.api.com

# 2. Load testing
python eval/benchmark.py --suite stress --duration 600 --max-concurrent 100

# 3. Post-deployment validation
python eval/quality_gate.py --threshold 0.85 --api-url https://prod.api.com
```

### Comparative Analysis

```bash
# Compare different model configurations
python eval/quality_gate.py --api-url http://model-a:8000 --output eval-model-a.json
python eval/quality_gate.py --api-url http://model-b:8000 --output eval-model-b.json

# Generate comparison report (custom script)
python eval/compare_models.py eval-model-a.json eval-model-b.json
```

## 🔍 Interpreting Results

### Quality Gate Results

- **Score ≥ 0.9**: Excellent quality, ready for production
- **Score 0.8-0.9**: Good quality, minor improvements possible
- **Score 0.7-0.8**: Acceptable quality, needs improvement
- **Score < 0.7**: Poor quality, significant improvements required

### Performance Benchmarks

- **Throughput**: Target >10 RPS for production workloads
- **P95 Latency**: Target <5s for interactive applications
- **Error Rate**: Target <1% for production systems
- **Stability**: Score >0.9 indicates good stress handling

### Common Issues and Solutions

1. **Low Keyword Coverage**
   - Review embedding model choice
   - Expand training data coverage
   - Adjust retrieval parameters

2. **Poor Semantic Similarity**
   - Fine-tune embedding model
   - Improve context preparation
   - Review query preprocessing

3. **High Response Times**
   - Optimize vector search
   - Implement caching
   - Scale infrastructure

4. **Low Technical Accuracy**
   - Improve domain-specific training
   - Add technical validation rules
   - Enhance LLM prompting

## 🛠 Advanced Features

### Custom Metrics

```python
# Add custom evaluation metric
from eval.metrics import RAGMetrics

class CustomRAGMetrics(RAGMetrics):
    def calculate_custom_metric(self, answer: str) -> float:
        # Your custom evaluation logic
        return score
```

### Extended Golden Dataset

```bash
# Add new test cases
echo '{"query": "New test query", "expected_keywords": [...]}' >> eval/golden_prompts.jsonl

# Validate dataset
python -c "import json; [json.loads(line) for line in open('eval/golden_prompts.jsonl')]"
```

### Performance Profiling

```python
# Profile specific components
import cProfile
cProfile.run('python eval/quality_gate.py', 'profile_results.prof')
```

## 📚 References

- [Evaluation Best Practices](https://docs.anthropic.com/evaluation)
- [RAG Evaluation Metrics](https://arxiv.org/abs/2401.15884)
- [Performance Testing Guidelines](https://martinfowler.com/articles/practical-test-pyramid.html)