#!/usr/bin/env python3
"""
Quality Gate Evaluation Script for Industrial Audio RAG

This script runs automated evaluation tests to ensure the RAG system
meets quality thresholds before deployment.

Usage:
    python eval/quality_gate.py --threshold 0.85
    python eval/quality_gate.py --config eval/config.yaml --output eval/results/
"""

import argparse
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import jsonschema
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data class for storing evaluation results."""

    query: str
    expected_keywords: List[str]
    response: Dict[str, Any]
    scores: Dict[str, float]
    passed: bool
    processing_time: float
    errors: List[str]


@dataclass
class QualityGateResults:
    """Data class for storing overall quality gate results."""

    total_queries: int
    passed_queries: int
    failed_queries: int
    overall_score: float
    individual_results: List[EvaluationResult]
    performance_metrics: Dict[str, float]
    threshold: float
    passed_quality_gate: bool


class RAGEvaluator:
    """Evaluator for Industrial Audio RAG system."""

    def __init__(self, api_base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

        # Set up authentication if API key provided
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def load_golden_dataset(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """Load golden dataset from JSONL file."""
        golden_queries = []

        try:
            with open(dataset_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        query_data = json.loads(line.strip())
                        golden_queries.append(query_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line {line_num}: {e}")

        except FileNotFoundError:
            logger.error(f"Golden dataset not found at {dataset_path}")
            raise

        logger.info(f"Loaded {len(golden_queries)} golden queries")
        return golden_queries

    def load_response_schema(self, schema_path: Path) -> Dict[str, Any]:
        """Load JSON schema for response validation."""
        try:
            with open(schema_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Schema file not found at {schema_path}, skipping validation")
            return {}

    def query_api(self, query: str) -> Dict[str, Any]:
        """Query the RAG API and return response."""
        url = urljoin(self.api_base_url, "/ask")
        params = {"q": query}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def validate_response_schema(
        self, response: Dict[str, Any], schema: Dict[str, Any]
    ) -> List[str]:
        """Validate response against JSON schema."""
        if not schema:
            return []

        errors = []
        try:
            jsonschema.validate(response, schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")

        return errors

    def calculate_keyword_coverage(self, answer: str, expected_keywords: List[str]) -> float:
        """Calculate how many expected keywords are present in the answer."""
        if not expected_keywords:
            return 1.0

        answer_lower = answer.lower()
        found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        return found_keywords / len(expected_keywords)

    def calculate_answer_quality_score(self, answer: str, expected_length: Dict[str, int]) -> float:
        """Calculate answer quality based on length and content."""
        if not answer.strip():
            return 0.0

        # Length score
        min_length = expected_length.get("min", 20)
        max_length = expected_length.get("max", 1000)
        answer_length = len(answer)

        if answer_length < min_length:
            length_score = answer_length / min_length * 0.8  # Penalty for too short
        elif answer_length > max_length:
            length_score = max_length / answer_length * 0.9  # Penalty for too long
        else:
            length_score = 1.0

        # Content quality heuristics
        sentences = answer.count(".") + answer.count("!") + answer.count("?")
        has_structure = sentences > 0
        has_numbers = any(char.isdigit() for char in answer)
        has_technical_terms = any(
            term in answer.lower()
            for term in ["hz", "frequency", "rms", "snr", "bearing", "valve", "gearbox"]
        )

        content_score = (
            0.3 * (1.0 if has_structure else 0.5)
            + 0.2 * (1.0 if has_numbers else 0.7)
            + 0.5 * (1.0 if has_technical_terms else 0.6)
        )

        return length_score * 0.4 + content_score * 0.6

    def calculate_performance_score(self, processing_time: float) -> float:
        """Calculate performance score based on processing time."""
        # Excellent: < 2s, Good: < 5s, Acceptable: < 10s, Poor: > 10s
        if processing_time < 2.0:
            return 1.0
        elif processing_time < 5.0:
            return 0.9
        elif processing_time < 10.0:
            return 0.7
        else:
            return 0.5

    def evaluate_single_query(
        self, query_data: Dict[str, Any], schema: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate a single query and return detailed results."""
        query = query_data["query"]
        expected_keywords = query_data.get("expected_keywords", [])
        expected_length = query_data.get("expected_answer_length", {"min": 20, "max": 500})

        start_time = time.time()
        errors = []

        try:
            # Query the API
            response = self.query_api(query)
            processing_time = time.time() - start_time

            # Validate schema
            schema_errors = self.validate_response_schema(response, schema)
            errors.extend(schema_errors)

            # Extract answer
            answer = response.get("answer", "")

            # Calculate scores
            keyword_score = self.calculate_keyword_coverage(answer, expected_keywords)
            quality_score = self.calculate_answer_quality_score(answer, expected_length)
            performance_score = self.calculate_performance_score(processing_time)

            # Overall score (weighted)
            overall_score = keyword_score * 0.4 + quality_score * 0.4 + performance_score * 0.2

            scores = {
                "keyword_coverage": keyword_score,
                "answer_quality": quality_score,
                "performance": performance_score,
                "overall": overall_score,
            }

            # Determine if passed (threshold applied later)
            passed = len(errors) == 0 and overall_score > 0.5  # Basic pass criteria

        except Exception as e:
            processing_time = time.time() - start_time
            errors.append(f"Evaluation error: {str(e)}")
            response = {}
            scores = {
                "keyword_coverage": 0.0,
                "answer_quality": 0.0,
                "performance": 0.0,
                "overall": 0.0,
            }
            passed = False

        return EvaluationResult(
            query=query,
            expected_keywords=expected_keywords,
            response=response,
            scores=scores,
            passed=passed,
            processing_time=processing_time,
            errors=errors,
        )

    def run_evaluation(
        self, golden_dataset: List[Dict[str, Any]], schema: Dict[str, Any], threshold: float = 0.8
    ) -> QualityGateResults:
        """Run full evaluation suite."""
        logger.info(
            f"Starting evaluation with {len(golden_dataset)} queries, threshold: {threshold}"
        )

        results = []
        for i, query_data in enumerate(golden_dataset, 1):
            logger.info(
                f"Evaluating query {i}/{len(golden_dataset)}: {query_data['query'][:50]}..."
            )
            result = self.evaluate_single_query(query_data, schema)
            results.append(result)

            # Brief pause to avoid overwhelming the API
            time.sleep(0.1)

        # Calculate overall metrics
        overall_scores = [r.scores["overall"] for r in results]
        overall_score = statistics.mean(overall_scores) if overall_scores else 0.0

        # Apply threshold
        passed_queries = sum(
            1 for r in results if r.scores["overall"] >= threshold and not r.errors
        )
        failed_queries = len(results) - passed_queries

        # Performance metrics
        processing_times = [r.processing_time for r in results]
        performance_metrics = {
            "avg_processing_time": statistics.mean(processing_times) if processing_times else 0.0,
            "p95_processing_time": statistics.quantiles(processing_times, n=20)[18]
            if len(processing_times) > 1
            else 0.0,
            "p99_processing_time": statistics.quantiles(processing_times, n=100)[98]
            if len(processing_times) > 1
            else 0.0,
            "max_processing_time": max(processing_times) if processing_times else 0.0,
        }

        passed_quality_gate = (passed_queries / len(results)) >= threshold if results else False

        return QualityGateResults(
            total_queries=len(results),
            passed_queries=passed_queries,
            failed_queries=failed_queries,
            overall_score=overall_score,
            individual_results=results,
            performance_metrics=performance_metrics,
            threshold=threshold,
            passed_quality_gate=passed_quality_gate,
        )

    def save_results(self, results: QualityGateResults, output_path: Path):
        """Save evaluation results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict for JSON serialization
        results_dict = asdict(results)

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Run quality gate evaluation")
    parser.add_argument("--threshold", type=float, default=0.8, help="Quality threshold (0.0-1.0)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/golden_prompts.jsonl"),
        help="Path to golden dataset",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("eval/evaluation_schema.json"),
        help="Path to response schema",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/evaluation_results.json"),
        help="Output path for results",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Exit on first failure")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RAGEvaluator(api_base_url=args.api_url, api_key=args.api_key)

    try:
        # Load data
        golden_dataset = evaluator.load_golden_dataset(args.dataset)
        schema = evaluator.load_response_schema(args.schema)

        # Run evaluation
        results = evaluator.run_evaluation(golden_dataset, schema, args.threshold)

        # Save results
        evaluator.save_results(results, args.output)

        # Print summary
        print(f"\n{'=' * 60}")
        print("QUALITY GATE EVALUATION RESULTS")
        print(f"{'=' * 60}")
        print(f"Total Queries: {results.total_queries}")
        print(f"Passed: {results.passed_queries}")
        print(f"Failed: {results.failed_queries}")
        print(f"Success Rate: {results.passed_queries / results.total_queries:.1%}")
        print(f"Overall Score: {results.overall_score:.3f}")
        print(f"Threshold: {results.threshold:.3f}")
        print(f"Quality Gate: {'✅ PASSED' if results.passed_quality_gate else '❌ FAILED'}")
        print("\nPerformance Metrics:")
        print(f"  Avg Processing Time: {results.performance_metrics['avg_processing_time']:.2f}s")
        print(f"  P95 Processing Time: {results.performance_metrics['p95_processing_time']:.2f}s")
        print(f"  Max Processing Time: {results.performance_metrics['max_processing_time']:.2f}s")

        # Exit with appropriate code
        exit_code = 0 if results.passed_quality_gate else 1

        if not results.passed_quality_gate:
            print(
                f"\n❌ Quality gate FAILED - Score {results.overall_score:.3f}"
                f" below threshold {results.threshold:.3f}"
            )

            # Show failed queries
            failed_results = [
                r
                for r in results.individual_results
                if r.scores["overall"] < args.threshold or r.errors
            ]
            if failed_results:
                print(f"\nFailed Queries ({len(failed_results)}):")
                for result in failed_results[:5]:  # Show first 5 failures
                    print(f"  - {result.query[:60]}... (Score: {result.scores['overall']:.3f})")
                    if result.errors:
                        for error in result.errors:
                            print(f"    Error: {error}")
        else:
            print("\n✅ Quality gate PASSED!")

        return exit_code

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
