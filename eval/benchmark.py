#!/usr/bin/env python3
"""
Comprehensive Benchmarking Framework for Industrial Audio RAG

This script runs performance benchmarks, stress tests, and comparative
evaluations to measure system performance under various conditions.

Usage:
    python eval/benchmark.py --suite performance
    python eval/benchmark.py --suite stress --duration 300
    python eval/benchmark.py --suite comparative --models model1,model2
"""

import argparse
import concurrent.futures
import json
import logging
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results."""

    test_name: str
    duration: float
    requests_sent: int
    requests_completed: int
    requests_failed: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_rps: float
    error_rate: float
    system_metrics: Dict[str, Any]
    test_config: Dict[str, Any]


@dataclass
class StressTestResult:
    """Data class for storing stress test results."""

    test_duration: float
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_throughput_rps: float
    peak_throughput_rps: float
    avg_response_time: float
    system_utilization: Dict[str, Any]
    breaking_point: Optional[int]
    stability_score: float


class SystemMonitor:
    """Monitor system resources during benchmarks."""

    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring system resources."""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        if not self.metrics:
            return {}

        # Aggregate metrics
        cpu_values = [m["cpu_percent"] for m in self.metrics]
        memory_values = [m["memory_percent"] for m in self.metrics]

        return {
            "cpu_percent": {
                "avg": statistics.mean(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "memory_percent": {
                "avg": statistics.mean(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
            },
            "samples": len(self.metrics),
        }

    def _monitor_loop(self, interval: float):
        """Monitor system resources in a loop."""
        while self.monitoring:
            try:
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_io": psutil.disk_io_counters()._asdict()
                    if psutil.disk_io_counters()
                    else {},
                    "network_io": psutil.net_io_counters()._asdict()
                    if psutil.net_io_counters()
                    else {},
                }
                self.metrics.append(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.warning(f"Error collecting system metrics: {e}")
                time.sleep(interval)


class RAGBenchmark:
    """Benchmarking framework for Industrial Audio RAG system."""

    def __init__(self, api_base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

        # Set up authentication if API key provided
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

        self.monitor = SystemMonitor()

    def load_test_queries(self, dataset_path: Path) -> List[str]:
        """Load test queries from dataset."""
        queries = []

        try:
            with open(dataset_path, "r") as f:
                for line in f:
                    try:
                        query_data = json.loads(line.strip())
                        queries.append(query_data["query"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        except FileNotFoundError:
            logger.warning(f"Dataset not found at {dataset_path}, using default queries")
            queries = [
                "Which bearing clips show high RMS values?",
                "Find valve recordings with dominant frequency above 500 Hz",
                "What is the average SNR of normal recordings?",
                "List all anomalous gearbox recordings",
                "Compare RMS between normal and anomalous states",
            ]

        logger.info(f"Loaded {len(queries)} test queries")
        return queries

    def make_request(self, query: str) -> Tuple[float, bool, Dict[str, Any]]:
        """Make a single API request and return timing and success info."""
        start_time = time.time()

        try:
            response = self.session.get(f"{self.api_base_url}/ask", params={"q": query}, timeout=30)
            response_time = time.time() - start_time

            if response.status_code == 200:
                return response_time, True, response.json()
            else:
                return response_time, False, {"error": f"HTTP {response.status_code}"}

        except Exception as e:
            response_time = time.time() - start_time
            return response_time, False, {"error": str(e)}

    def run_performance_benchmark(
        self, queries: List[str], iterations: int = 10
    ) -> BenchmarkResult:
        """Run performance benchmark with sequential requests."""
        logger.info(f"Running performance benchmark: {iterations} iterations")

        self.monitor.start_monitoring()
        start_time = time.time()

        response_times = []
        successful_requests = 0
        failed_requests = 0

        for i in range(iterations):
            query = queries[i % len(queries)]
            response_time, success, _ = self.make_request(query)

            response_times.append(response_time)
            if success:
                successful_requests += 1
            else:
                failed_requests += 1

            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{iterations} requests")

        total_duration = time.time() - start_time
        system_metrics = self.monitor.stop_monitoring()

        # Calculate statistics
        throughput = successful_requests / total_duration if total_duration > 0 else 0
        error_rate = failed_requests / iterations if iterations > 0 else 0

        return BenchmarkResult(
            test_name="performance_benchmark",
            duration=total_duration,
            requests_sent=iterations,
            requests_completed=successful_requests,
            requests_failed=failed_requests,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            p50_response_time=statistics.median(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18]
            if len(response_times) > 1
            else 0,
            p99_response_time=statistics.quantiles(response_times, n=100)[98]
            if len(response_times) > 1
            else 0,
            throughput_rps=throughput,
            error_rate=error_rate,
            system_metrics=system_metrics,
            test_config={"iterations": iterations, "queries": len(queries)},
        )

    def run_stress_test(
        self, queries: List[str], duration: int = 60, max_concurrent: int = 50
    ) -> StressTestResult:
        """Run stress test with increasing load."""
        logger.info(
            f"Running stress test: {duration}s duration, max {max_concurrent} concurrent users"
        )

        self.monitor.start_monitoring()
        start_time = time.time()

        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []
        throughput_samples = []
        breaking_point = None

        # Gradually increase concurrent users
        for concurrent_users in range(1, max_concurrent + 1, 5):
            batch_start = time.time()
            batch_requests = 0
            batch_successful = 0

            # Run for a portion of total duration
            batch_duration = min(10, duration / (max_concurrent // 5))
            end_time = batch_start + batch_duration

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = []

                while time.time() < end_time:
                    query = queries[batch_requests % len(queries)]
                    future = executor.submit(self.make_request, query)
                    futures.append(future)
                    batch_requests += 1

                    # Small delay to avoid overwhelming
                    time.sleep(0.01)

                # Collect results
                for future in concurrent.futures.as_completed(futures, timeout=30):
                    try:
                        response_time, success, _ = future.result()
                        response_times.append(response_time)
                        if success:
                            batch_successful += 1
                        else:
                            failed_requests += 1
                    except Exception as e:
                        logger.warning(f"Request failed: {e}")
                        failed_requests += 1

            total_requests += batch_requests
            successful_requests += batch_successful

            # Calculate throughput for this batch
            batch_throughput = batch_successful / batch_duration
            throughput_samples.append(batch_throughput)

            # Check for breaking point (>10% error rate)
            batch_error_rate = (
                (batch_requests - batch_successful) / batch_requests if batch_requests > 0 else 0
            )
            if batch_error_rate > 0.1 and breaking_point is None:
                breaking_point = concurrent_users
                logger.warning(f"Breaking point detected at {concurrent_users} concurrent users")

            logger.info(
                f"Concurrent users: {concurrent_users}, Throughput: {batch_throughput:.2f} RPS, "
                f"Error rate: {batch_error_rate:.1%}"
            )

            # Stop if we've exceeded the total duration
            if time.time() - start_time >= duration:
                break

        total_duration = time.time() - start_time
        system_metrics = self.monitor.stop_monitoring()

        # Calculate stability score (1.0 = no breaking point, decreases based on when it broke)
        stability_score = (
            1.0 if breaking_point is None else max(0, (breaking_point - 1) / max_concurrent)
        )

        return StressTestResult(
            test_duration=total_duration,
            concurrent_users=max_concurrent,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_throughput_rps=statistics.mean(throughput_samples) if throughput_samples else 0,
            peak_throughput_rps=max(throughput_samples) if throughput_samples else 0,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            system_utilization=system_metrics,
            breaking_point=breaking_point,
            stability_score=stability_score,
        )

    def run_latency_benchmark(self, queries: List[str], samples: int = 100) -> Dict[str, Any]:
        """Run detailed latency analysis."""
        logger.info(f"Running latency benchmark: {samples} samples")

        response_times = []

        for i in range(samples):
            query = queries[i % len(queries)]
            response_time, success, _ = self.make_request(query)

            if success:
                response_times.append(response_time)

            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{samples} latency samples")

        if not response_times:
            return {"error": "No successful requests"}

        # Calculate detailed percentiles
        percentiles = [50, 75, 90, 95, 99, 99.9]
        percentile_values = statistics.quantiles(response_times, n=1000)

        return {
            "sample_count": len(response_times),
            "mean": statistics.mean(response_times),
            "median": statistics.median(response_times),
            "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "min": min(response_times),
            "max": max(response_times),
            "percentiles": {
                f"p{p}": percentile_values[int(p * 10 - 1)]
                if p * 10 - 1 < len(percentile_values)
                else max(response_times)
                for p in percentiles
            },
        }

    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save benchmark results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Run RAG system benchmarks")
    parser.add_argument(
        "--suite",
        choices=["performance", "stress", "latency", "all"],
        default="performance",
        help="Benchmark suite to run",
    )
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/golden_prompts.jsonl"),
        help="Path to test queries dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/benchmark_results.json"),
        help="Output path for results",
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of iterations for performance test"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Duration in seconds for stress test"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=20, help="Maximum concurrent users for stress test"
    )
    parser.add_argument(
        "--latency-samples", type=int, default=100, help="Number of samples for latency test"
    )

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = RAGBenchmark(api_base_url=args.api_url, api_key=args.api_key)

    try:
        # Load test queries
        queries = benchmark.load_test_queries(args.dataset)

        results = {}

        # Run selected benchmark suite
        if args.suite in ["performance", "all"]:
            logger.info("Running performance benchmark...")
            perf_result = benchmark.run_performance_benchmark(queries, args.iterations)
            results["performance"] = asdict(perf_result)

        if args.suite in ["stress", "all"]:
            logger.info("Running stress test...")
            stress_result = benchmark.run_stress_test(queries, args.duration, args.max_concurrent)
            results["stress"] = asdict(stress_result)

        if args.suite in ["latency", "all"]:
            logger.info("Running latency benchmark...")
            latency_result = benchmark.run_latency_benchmark(queries, args.latency_samples)
            results["latency"] = latency_result

        # Add metadata
        results["metadata"] = {
            "timestamp": time.time(),
            "api_url": args.api_url,
            "test_config": {
                "suite": args.suite,
                "iterations": args.iterations,
                "duration": args.duration,
                "max_concurrent": args.max_concurrent,
                "latency_samples": args.latency_samples,
            },
        }

        # Save results
        benchmark.save_results(results, args.output)

        # Print summary
        print(f"\n{'=' * 60}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'=' * 60}")

        if "performance" in results:
            perf = results["performance"]
            print("\nPerformance Benchmark:")
            print(f"  Requests: {perf['requests_completed']}/{perf['requests_sent']}")
            print(f"  Throughput: {perf['throughput_rps']:.2f} RPS")
            print(f"  Avg Response Time: {perf['avg_response_time']:.3f}s")
            print(f"  P95 Response Time: {perf['p95_response_time']:.3f}s")
            print(f"  Error Rate: {perf['error_rate']:.1%}")

        if "stress" in results:
            stress = results["stress"]
            print("\nStress Test:")
            print(f"  Peak Throughput: {stress['peak_throughput_rps']:.2f} RPS")
            print(f"  Avg Throughput: {stress['avg_throughput_rps']:.2f} RPS")
            print(f"  Breaking Point: {stress['breaking_point'] or 'Not reached'}")
            print(f"  Stability Score: {stress['stability_score']:.2f}")
            print(f"  Success Rate: {stress['successful_requests'] / stress['total_requests']:.1%}")

        if "latency" in results:
            latency = results["latency"]
            print("\nLatency Analysis:")
            print(f"  Mean: {latency['mean']:.3f}s")
            print(f"  P50: {latency['percentiles']['p50']:.3f}s")
            print(f"  P95: {latency['percentiles']['p95']:.3f}s")
            print(f"  P99: {latency['percentiles']['p99']:.3f}s")
            print(f"  Max: {latency['max']:.3f}s")

        print(f"\nFull results saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
