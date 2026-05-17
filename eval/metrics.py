#!/usr/bin/env python3
"""
Evaluation Metrics and Scoring Framework for Industrial Audio RAG

This module provides comprehensive metrics for evaluating RAG system performance
including semantic similarity, retrieval accuracy, and answer quality.

Usage:
    from eval.metrics import RAGMetrics

    metrics = RAGMetrics()
    score = metrics.evaluate_response(query, expected, actual)
"""

import logging
import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False
    logging.warning(
        "Advanced metrics not available. Install sentence-transformers and"
        " scikit-learn for full functionality."
    )

try:
    import nltk

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Some text metrics will be limited.")


@dataclass
class EvaluationScore:
    """Data class for storing evaluation scores."""

    overall_score: float
    keyword_coverage: float
    semantic_similarity: float
    answer_quality: float
    retrieval_accuracy: float
    response_completeness: float
    factual_consistency: float
    length_appropriateness: float
    technical_accuracy: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "keyword_coverage": self.keyword_coverage,
            "semantic_similarity": self.semantic_similarity,
            "answer_quality": self.answer_quality,
            "retrieval_accuracy": self.retrieval_accuracy,
            "response_completeness": self.response_completeness,
            "factual_consistency": self.factual_consistency,
            "length_appropriateness": self.length_appropriateness,
            "technical_accuracy": self.technical_accuracy,
        }


class RAGMetrics:
    """Comprehensive metrics for evaluating RAG system performance."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize metrics calculator."""
        self.embedding_model_name = embedding_model
        self.embedder = None

        # Initialize embedding model if available
        if ADVANCED_METRICS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(embedding_model)
                logging.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logging.warning(f"Failed to load embedding model: {e}")

        # Initialize NLTK if available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                logging.info("Downloading NLTK punkt tokenizer...")
                nltk.download("punkt", quiet=True)

    def calculate_keyword_coverage(self, answer: str, expected_keywords: List[str]) -> float:
        """Calculate how many expected keywords are present in the answer."""
        if not expected_keywords:
            return 1.0

        answer_lower = answer.lower()
        found_keywords = []

        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            # Check for exact match or partial match
            if keyword_lower in answer_lower:
                found_keywords.append(keyword)
            # Check for word boundaries to avoid false positives
            elif re.search(r"\b" + re.escape(keyword_lower) + r"\b", answer_lower):
                found_keywords.append(keyword)

        coverage = len(found_keywords) / len(expected_keywords)
        return min(coverage, 1.0)

    def calculate_semantic_similarity(
        self, query: str, answer: str, expected_context: str = ""
    ) -> float:
        """Calculate semantic similarity between answer and expected context."""
        if not ADVANCED_METRICS_AVAILABLE or self.embedder is None:
            # Fallback to simple keyword overlap
            return self._calculate_simple_similarity(answer, expected_context)

        try:
            # Combine query and expected context for comparison
            reference_text = f"{query} {expected_context}".strip()
            if not reference_text:
                reference_text = query

            # Get embeddings
            answer_embedding = self.embedder.encode([answer])
            reference_embedding = self.embedder.encode([reference_text])

            # Calculate cosine similarity
            similarity = cosine_similarity(answer_embedding, reference_embedding)[0][0]
            return float(similarity)

        except Exception as e:
            logging.warning(f"Error calculating semantic similarity: {e}")
            return self._calculate_simple_similarity(answer, expected_context)

    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity based on word overlap."""
        if not text2:
            return 0.5  # Neutral score if no reference

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def calculate_answer_quality(self, answer: str, expected_length: Dict[str, int]) -> float:
        """Calculate answer quality based on multiple factors."""
        if not answer.strip():
            return 0.0

        scores = []

        # 1. Length appropriateness
        length_score = self._calculate_length_score(answer, expected_length)
        scores.append(length_score * 0.2)

        # 2. Structure and coherence
        structure_score = self._calculate_structure_score(answer)
        scores.append(structure_score * 0.3)

        # 3. Information density
        density_score = self._calculate_information_density(answer)
        scores.append(density_score * 0.2)

        # 4. Technical accuracy (domain-specific)
        technical_score = self._calculate_technical_accuracy(answer)
        scores.append(technical_score * 0.3)

        return sum(scores)

    def _calculate_length_score(self, answer: str, expected_length: Dict[str, int]) -> float:
        """Score based on answer length appropriateness."""
        min_length = expected_length.get("min", 20)
        max_length = expected_length.get("max", 1000)
        answer_length = len(answer)

        if answer_length < min_length:
            return answer_length / min_length * 0.8  # Penalty for too short
        elif answer_length > max_length:
            return max_length / answer_length * 0.9  # Penalty for too long
        else:
            # Optimal range
            optimal_length = (min_length + max_length) / 2
            distance_from_optimal = abs(answer_length - optimal_length) / optimal_length
            return max(0.9, 1.0 - distance_from_optimal * 0.1)

    def _calculate_structure_score(self, answer: str) -> float:
        """Score based on answer structure and coherence."""
        # Count sentences
        sentences = len([s for s in re.split(r"[.!?]+", answer) if s.strip()])

        # Count paragraphs
        paragraphs = len([p for p in answer.split("\n\n") if p.strip()])

        # Check for lists or structured content
        has_lists = bool(re.search(r"[-*•]\s|^\d+\.", answer, re.MULTILINE))

        # Check for proper punctuation
        has_punctuation = any(char in answer for char in ".!?")

        # Scoring
        structure_score = 0.0

        # Sentence structure (0-0.4)
        if sentences == 0:
            structure_score += 0.0
        elif sentences == 1:
            structure_score += 0.2
        else:
            structure_score += min(0.4, sentences * 0.1)

        # Paragraphs (0-0.2)
        if paragraphs > 1:
            structure_score += 0.2

        # Lists (0-0.2)
        if has_lists:
            structure_score += 0.2

        # Punctuation (0-0.2)
        if has_punctuation:
            structure_score += 0.2

        return min(1.0, structure_score)

    def _calculate_information_density(self, answer: str) -> float:
        """Score based on information density and specificity."""
        # Count numbers (indicates specific information)
        numbers = len(re.findall(r"\d+\.?\d*", answer))

        # Count technical terms
        technical_terms = len(
            re.findall(
                r"\b(?:Hz|RMS|SNR|dB|frequency|bearing|valve|gearbox|train|fan|slider|anomalous|normal|section|domain|target|source)\b",
                answer,
                re.IGNORECASE,
            )
        )

        # Count unique words (vocabulary diversity)
        words = answer.lower().split()
        unique_words = len(set(words))

        # Scoring
        word_count = len(words)
        if word_count == 0:
            return 0.0

        # Numbers density (0-0.3)
        number_density = min(0.3, numbers / max(1, word_count) * 10)

        # Technical terms density (0-0.4)
        tech_density = min(0.4, technical_terms / max(1, word_count) * 5)

        # Vocabulary diversity (0-0.3)
        diversity = min(0.3, unique_words / max(1, word_count))

        return number_density + tech_density + diversity

    def _calculate_technical_accuracy(self, answer: str) -> float:
        """Score based on technical accuracy for industrial audio domain."""
        score = 0.0

        # Check for domain-appropriate terminology
        audio_terms = ["frequency", "hz", "rms", "snr", "signal", "noise", "amplitude", "spectrum"]
        industrial_terms = ["bearing", "valve", "gearbox", "train", "fan", "slider", "machine"]
        condition_terms = ["normal", "anomalous", "fault", "wear", "vibration"]

        answer_lower = answer.lower()

        # Audio terminology (0-0.3)
        audio_matches = sum(1 for term in audio_terms if term in answer_lower)
        score += min(0.3, audio_matches / len(audio_terms))

        # Industrial terminology (0-0.4)
        industrial_matches = sum(1 for term in industrial_terms if term in answer_lower)
        score += min(0.4, industrial_matches / len(industrial_terms))

        # Condition terminology (0-0.3)
        condition_matches = sum(1 for term in condition_terms if term in answer_lower)
        score += min(0.3, condition_matches / len(condition_terms))

        return min(1.0, score)

    def calculate_retrieval_accuracy(
        self, expected_file_count: Dict[str, int], actual_files_mentioned: int
    ) -> float:
        """Score based on retrieval accuracy."""
        min_expected = expected_file_count.get("min", 1)
        max_expected = expected_file_count.get("max", 10)

        if actual_files_mentioned < min_expected:
            return actual_files_mentioned / min_expected * 0.8
        elif actual_files_mentioned > max_expected:
            return max_expected / actual_files_mentioned * 0.9
        else:
            return 1.0

    def calculate_response_completeness(
        self, query: str, answer: str, expected_keywords: List[str]
    ) -> float:
        """Score based on how completely the answer addresses the query."""
        # Extract query intent indicators
        query_lower = query.lower()

        completeness_indicators = {
            "what": ["what", "which", "how many", "how much"],
            "where": ["where", "section", "location"],
            "why": ["why", "reason", "cause"],
            "how": ["how", "method", "process"],
            "compare": ["compare", "difference", "versus", "vs"],
            "list": ["list", "show", "find all", "enumerate"],
            "analyze": ["analyze", "examine", "evaluate", "assess"],
        }

        # Identify query type
        query_types = []
        for q_type, indicators in completeness_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                query_types.append(q_type)

        if not query_types:
            query_types = ["general"]  # Default type

        # Check answer completeness based on query type
        answer_lower = answer.lower()
        completeness_score = 0.0

        for q_type in query_types:
            if q_type == "what":
                # Should provide specific information
                has_specifics = bool(re.search(r"\d+|specific|particular", answer_lower))
                completeness_score += 0.3 if has_specifics else 0.1

            elif q_type == "compare":
                # Should mention both sides of comparison
                has_comparison = any(
                    word in answer_lower
                    for word in ["both", "while", "whereas", "compared", "than"]
                )
                completeness_score += 0.4 if has_comparison else 0.1

            elif q_type == "list":
                # Should provide multiple items
                has_multiple = bool(re.search(r"[;,]\s|\n-|\n\d+\.", answer))
                completeness_score += 0.4 if has_multiple else 0.2

            else:
                # General completeness check
                completeness_score += 0.3

        # Keyword coverage bonus
        keyword_coverage = self.calculate_keyword_coverage(answer, expected_keywords)
        completeness_score += keyword_coverage * 0.3

        return min(1.0, completeness_score)

    def calculate_factual_consistency(self, answer: str) -> float:
        """Score based on factual consistency and lack of contradictions."""
        # Check for contradictory statements
        contradictions = 0

        # Look for conflicting numbers
        numbers = re.findall(r"\d+\.?\d*", answer)
        if len(set(numbers)) != len(numbers):
            # Repeated numbers might indicate inconsistency
            contradictions += 0.1

        # Look for conflicting qualitative statements
        conflicting_pairs = [
            (["high", "large", "big"], ["low", "small", "little"]),
            (["normal", "healthy"], ["anomalous", "fault", "abnormal"]),
            (["increase", "rise"], ["decrease", "fall"]),
        ]

        for positive, negative in conflicting_pairs:
            has_positive = any(word in answer.lower() for word in positive)
            has_negative = any(word in answer.lower() for word in negative)
            if has_positive and has_negative:
                contradictions += 0.2

        # Base score minus contradictions
        consistency_score = 1.0 - min(1.0, contradictions)

        return max(0.0, consistency_score)

    def evaluate_response(
        self, query: str, expected_data: Dict[str, Any], actual_response: Dict[str, Any]
    ) -> EvaluationScore:
        """Comprehensive evaluation of a RAG response."""
        answer = actual_response.get("answer", "")
        expected_keywords = expected_data.get("expected_keywords", [])
        expected_length = expected_data.get("expected_answer_length", {"min": 20, "max": 500})
        expected_file_count = expected_data.get("expected_file_count", {"min": 1, "max": 10})

        # Calculate individual metrics
        keyword_coverage = self.calculate_keyword_coverage(answer, expected_keywords)
        semantic_similarity = self.calculate_semantic_similarity(query, answer)
        answer_quality = self.calculate_answer_quality(answer, expected_length)

        # Estimate files mentioned (simple heuristic)
        files_mentioned = len(re.findall(r"file|recording|clip|sample", answer, re.IGNORECASE))
        retrieval_accuracy = self.calculate_retrieval_accuracy(expected_file_count, files_mentioned)

        response_completeness = self.calculate_response_completeness(
            query, answer, expected_keywords
        )
        factual_consistency = self.calculate_factual_consistency(answer)
        length_appropriateness = self._calculate_length_score(answer, expected_length)
        technical_accuracy = self._calculate_technical_accuracy(answer)

        # Calculate weighted overall score
        overall_score = (
            keyword_coverage * 0.15
            + semantic_similarity * 0.20
            + answer_quality * 0.25
            + retrieval_accuracy * 0.10
            + response_completeness * 0.15
            + factual_consistency * 0.10
            + technical_accuracy * 0.05
        )

        return EvaluationScore(
            overall_score=overall_score,
            keyword_coverage=keyword_coverage,
            semantic_similarity=semantic_similarity,
            answer_quality=answer_quality,
            retrieval_accuracy=retrieval_accuracy,
            response_completeness=response_completeness,
            factual_consistency=factual_consistency,
            length_appropriateness=length_appropriateness,
            technical_accuracy=technical_accuracy,
        )

    def batch_evaluate(
        self, evaluations: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Evaluate multiple responses and return aggregated metrics."""
        scores = []
        individual_results = []

        for query, expected, actual in evaluations:
            score = self.evaluate_response(query, expected, actual)
            scores.append(score)
            individual_results.append({"query": query, "scores": score.to_dict()})

        if not scores:
            return {"error": "No evaluations provided"}

        # Aggregate statistics
        metrics = {}
        for metric_name in [
            "overall_score",
            "keyword_coverage",
            "semantic_similarity",
            "answer_quality",
            "retrieval_accuracy",
            "response_completeness",
            "factual_consistency",
            "length_appropriateness",
            "technical_accuracy",
        ]:
            values = [getattr(score, metric_name) for score in scores]
            metrics[metric_name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "q25": statistics.quantiles(values, n=4)[0] if len(values) > 1 else values[0],
                "q75": statistics.quantiles(values, n=4)[2] if len(values) > 1 else values[0],
            }

        return {
            "aggregated_metrics": metrics,
            "individual_results": individual_results,
            "summary": {
                "total_evaluations": len(scores),
                "average_overall_score": metrics["overall_score"]["mean"],
                "score_distribution": {
                    "excellent": sum(1 for s in scores if s.overall_score >= 0.9),
                    "good": sum(1 for s in scores if 0.7 <= s.overall_score < 0.9),
                    "fair": sum(1 for s in scores if 0.5 <= s.overall_score < 0.7),
                    "poor": sum(1 for s in scores if s.overall_score < 0.5),
                },
            },
        }


def main():
    """Example usage of RAG metrics."""
    metrics = RAGMetrics()

    # Example evaluation
    query = "Which bearing clips in section 00 show dominant frequency above 900 Hz?"
    expected = {
        "expected_keywords": ["bearing", "section", "00", "900", "Hz", "frequency"],
        "expected_answer_length": {"min": 50, "max": 300},
        "expected_file_count": {"min": 1, "max": 10},
    }
    actual = {
        "answer": (
            "Based on the audio analysis, I found 3 bearing recordings in section 00 with"
            " dominant frequencies above 900 Hz. These include"
            " bearing_00_source_train_normal_000145.wav with a peak at 1024 Hz,"
            " bearing_00_target_train_normal_000201.wav at 950 Hz, and"
            " bearing_00_source_train_anomalous_000067.wav at 1156 Hz."
            " The high frequencies might indicate bearing wear or misalignment issues."
        )
    }

    score = metrics.evaluate_response(query, expected, actual)

    print("Evaluation Results:")
    print(f"Overall Score: {score.overall_score:.3f}")
    print(f"Keyword Coverage: {score.keyword_coverage:.3f}")
    print(f"Semantic Similarity: {score.semantic_similarity:.3f}")
    print(f"Answer Quality: {score.answer_quality:.3f}")
    print(f"Technical Accuracy: {score.technical_accuracy:.3f}")


if __name__ == "__main__":
    main()
