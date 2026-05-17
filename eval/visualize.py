#!/usr/bin/env python3
"""
Evaluation Visualization and Reporting for Industrial Audio RAG

This script generates comprehensive reports and visualizations from evaluation results.

Usage:
    python eval/visualize.py --input eval/results/evaluation_results.json
    python eval/visualize.py --benchmark eval/results/benchmark_results.json --output reports/
"""

import argparse
import json
import logging
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Try to import optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning(
        "Visualization libraries not available. Install matplotlib, seaborn, and pandas"
        " for full functionality."
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate comprehensive evaluation reports and visualizations."""

    def __init__(self, output_dir: Path = Path("eval/reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if VISUALIZATION_AVAILABLE:
            # Set visualization style
            plt.style.use("seaborn-v0_8")
            sns.set_palette("husl")

    def load_results(self, results_path: Path) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        try:
            with open(results_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Results file not found: {results_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            raise

    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive quality evaluation report."""
        report_lines = []

        # Header
        report_lines.append("# Industrial Audio RAG - Quality Evaluation Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Executive Summary
        report_lines.append("## Executive Summary")
        overall_score = results.get("overall_score", 0)
        passed = results.get("passed_quality_gate", False)
        threshold = results.get("threshold", 0.8)

        report_lines.append(f"- **Overall Score**: {overall_score:.3f}")
        report_lines.append(f"- **Quality Gate**: {'✅ PASSED' if passed else '❌ FAILED'}")
        report_lines.append(f"- **Threshold**: {threshold:.3f}")
        report_lines.append(f"- **Total Queries**: {results.get('total_queries', 0)}")
        report_lines.append(
            f"- **Success Rate**: {results.get('passed_queries', 0)}"
            f"/{results.get('total_queries', 0)}"
            f" ({results.get('passed_queries', 0) / max(1, results.get('total_queries', 1)):.1%})"
        )
        report_lines.append("")

        # Performance Metrics
        perf_metrics = results.get("performance_metrics", {})
        if perf_metrics:
            report_lines.append("## Performance Metrics")
            report_lines.append(
                f"- **Average Processing Time**: {perf_metrics.get('avg_processing_time', 0):.2f}s"
            )
            report_lines.append(
                f"- **P95 Processing Time**: {perf_metrics.get('p95_processing_time', 0):.2f}s"
            )
            report_lines.append(
                f"- **P99 Processing Time**: {perf_metrics.get('p99_processing_time', 0):.2f}s"
            )
            report_lines.append(
                f"- **Max Processing Time**: {perf_metrics.get('max_processing_time', 0):.2f}s"
            )
            report_lines.append("")

        # Individual Results Analysis
        individual_results = results.get("individual_results", [])
        if individual_results:
            report_lines.append("## Query Analysis")

            # Score distribution
            scores = [r["scores"]["overall"] for r in individual_results]
            report_lines.append("- **Score Statistics**:")
            report_lines.append(f"  - Mean: {statistics.mean(scores):.3f}")
            report_lines.append(f"  - Median: {statistics.median(scores):.3f}")
            report_lines.append(
                f"  - Std Dev: {statistics.stdev(scores) if len(scores) > 1 else 0:.3f}"
            )
            report_lines.append(f"  - Min: {min(scores):.3f}")
            report_lines.append(f"  - Max: {max(scores):.3f}")
            report_lines.append("")

            # Failed queries
            failed_queries = [
                r
                for r in individual_results
                if r["scores"]["overall"] < threshold or r.get("errors")
            ]
            if failed_queries:
                report_lines.append(f"### Failed Queries ({len(failed_queries)})")
                for i, result in enumerate(failed_queries[:10], 1):  # Show top 10 failures
                    query = (
                        result["query"][:100] + "..."
                        if len(result["query"]) > 100
                        else result["query"]
                    )
                    score = result["scores"]["overall"]
                    report_lines.append(f"{i}. **Score: {score:.3f}** - {query}")

                    # Show specific issues
                    scores = result["scores"]
                    issues = []
                    if scores.get("keyword_coverage", 1) < 0.5:
                        issues.append("Poor keyword coverage")
                    if scores.get("answer_quality", 1) < 0.5:
                        issues.append("Low answer quality")
                    if scores.get("semantic_similarity", 1) < 0.5:
                        issues.append("Poor semantic similarity")

                    if issues:
                        report_lines.append(f"   Issues: {', '.join(issues)}")

                    if result.get("errors"):
                        report_lines.append(f"   Errors: {'; '.join(result['errors'])}")

                    report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        if not passed:
            report_lines.append("### Critical Actions Required")
            report_lines.append("- Review failed queries and improve response quality")
            report_lines.append("- Consider adjusting embedding model or search parameters")
            report_lines.append("- Validate training data coverage for identified weak areas")
        else:
            report_lines.append("### Optimization Opportunities")
            report_lines.append("- Monitor performance metrics for degradation")
            report_lines.append("- Consider expanding test coverage for edge cases")

        if perf_metrics.get("avg_processing_time", 0) > 5:
            report_lines.append("- Optimize response time (currently > 5s average)")

        report_lines.append("- Implement continuous monitoring of quality metrics")
        report_lines.append("- Set up alerting for quality gate failures")
        report_lines.append("")

        return "\n".join(report_lines)

    def generate_benchmark_report(self, results: Dict[str, Any]) -> str:
        """Generate benchmark performance report."""
        report_lines = []

        # Header
        report_lines.append("# Industrial Audio RAG - Benchmark Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Performance Benchmark
        if "performance" in results:
            perf = results["performance"]
            report_lines.append("## Performance Benchmark")
            report_lines.append(f"- **Throughput**: {perf['throughput_rps']:.2f} RPS")
            report_lines.append(f"- **Average Response Time**: {perf['avg_response_time']:.3f}s")
            report_lines.append(f"- **P95 Response Time**: {perf['p95_response_time']:.3f}s")
            report_lines.append(f"- **P99 Response Time**: {perf['p99_response_time']:.3f}s")
            report_lines.append(f"- **Error Rate**: {perf['error_rate']:.1%}")
            report_lines.append(
                f"- **Requests Completed**: {perf['requests_completed']}/{perf['requests_sent']}"
            )
            report_lines.append("")

        # Stress Test
        if "stress" in results:
            stress = results["stress"]
            report_lines.append("## Stress Test Results")
            report_lines.append(f"- **Peak Throughput**: {stress['peak_throughput_rps']:.2f} RPS")
            report_lines.append(f"- **Average Throughput**: {stress['avg_throughput_rps']:.2f} RPS")
            report_lines.append(
                f"- **Breaking Point**: {stress['breaking_point'] or 'Not reached'}"
            )
            report_lines.append(f"- **Stability Score**: {stress['stability_score']:.2f}")
            report_lines.append(f"- **Total Requests**: {stress['total_requests']}")
            report_lines.append(
                "- **Success Rate**:"
                f" {stress['successful_requests'] / stress['total_requests']:.1%}"
            )
            report_lines.append("")

        # Latency Analysis
        if "latency" in results:
            latency = results["latency"]
            report_lines.append("## Latency Analysis")
            report_lines.append(f"- **Mean Latency**: {latency['mean']:.3f}s")
            report_lines.append(f"- **Median (P50)**: {latency['percentiles']['p50']:.3f}s")
            report_lines.append(f"- **P95**: {latency['percentiles']['p95']:.3f}s")
            report_lines.append(f"- **P99**: {latency['percentiles']['p99']:.3f}s")
            report_lines.append(f"- **P99.9**: {latency['percentiles']['p99.9']:.3f}s")
            report_lines.append(f"- **Maximum**: {latency['max']:.3f}s")
            report_lines.append(f"- **Standard Deviation**: {latency['std_dev']:.3f}s")
            report_lines.append("")

        return "\n".join(report_lines)

    def create_quality_visualizations(self, results: Dict[str, Any]) -> List[Path]:
        """Create visualizations for quality evaluation results."""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available, skipping charts")
            return []

        plots_created = []
        individual_results = results.get("individual_results", [])

        if not individual_results:
            logger.warning("No individual results found for visualization")
            return plots_created

        # Extract data for plotting
        scores_data = []
        for result in individual_results:
            scores = result["scores"]
            scores_data.append(
                {
                    "query": result["query"][:50] + "...",
                    "overall": scores["overall"],
                    "keyword_coverage": scores["keyword_coverage"],
                    "answer_quality": scores["answer_quality"],
                    "semantic_similarity": scores["semantic_similarity"],
                    "processing_time": result["processing_time"],
                }
            )

        df = pd.DataFrame(scores_data)

        # 1. Score Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Overall score histogram
        axes[0, 0].hist(df["overall"], bins=20, alpha=0.7, color="skyblue")
        axes[0, 0].axvline(
            results.get("threshold", 0.8), color="red", linestyle="--", label="Threshold"
        )
        axes[0, 0].set_title("Overall Score Distribution")
        axes[0, 0].set_xlabel("Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].legend()

        # Metric comparison
        metrics = ["keyword_coverage", "answer_quality", "semantic_similarity"]
        metric_means = [df[metric].mean() for metric in metrics]
        axes[0, 1].bar(metrics, metric_means, color=["lightcoral", "lightgreen", "lightblue"])
        axes[0, 1].set_title("Average Metric Scores")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Processing time vs score
        axes[1, 0].scatter(df["processing_time"], df["overall"], alpha=0.6)
        axes[1, 0].set_title("Processing Time vs Overall Score")
        axes[1, 0].set_xlabel("Processing Time (s)")
        axes[1, 0].set_ylabel("Overall Score")

        # Score correlation heatmap
        corr_data = df[
            ["overall", "keyword_coverage", "answer_quality", "semantic_similarity"]
        ].corr()
        sns.heatmap(corr_data, annot=True, cmap="coolwarm", center=0, ax=axes[1, 1])
        axes[1, 1].set_title("Metric Correlations")

        plt.tight_layout()
        plot_path = self.output_dir / "quality_evaluation_charts.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots_created.append(plot_path)

        # 2. Detailed metrics radar chart for top/bottom queries
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw=dict(projection="polar"))

        # Top performing query
        top_query = df.loc[df["overall"].idxmax()]
        categories = ["Keyword\nCoverage", "Answer\nQuality", "Semantic\nSimilarity"]
        values_top = [
            top_query["keyword_coverage"],
            top_query["answer_quality"],
            top_query["semantic_similarity"],
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values_top += values_top[:1]  # Complete the circle
        angles += angles[:1]

        axes[0].plot(
            angles, values_top, "o-", linewidth=2, label=f"Score: {top_query['overall']:.3f}"
        )
        axes[0].fill(angles, values_top, alpha=0.25)
        axes[0].set_xticks(angles[:-1])
        axes[0].set_xticklabels(categories)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Best Performing Query", pad=20)
        axes[0].legend()

        # Worst performing query
        worst_query = df.loc[df["overall"].idxmin()]
        values_worst = [
            worst_query["keyword_coverage"],
            worst_query["answer_quality"],
            worst_query["semantic_similarity"],
        ]
        values_worst += values_worst[:1]

        axes[1].plot(
            angles,
            values_worst,
            "o-",
            linewidth=2,
            color="red",
            label=f"Score: {worst_query['overall']:.3f}",
        )
        axes[1].fill(angles, values_worst, alpha=0.25, color="red")
        axes[1].set_xticks(angles[:-1])
        axes[1].set_xticklabels(categories)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Worst Performing Query", pad=20)
        axes[1].legend()

        plt.tight_layout()
        radar_path = self.output_dir / "performance_radar_charts.png"
        plt.savefig(radar_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots_created.append(radar_path)

        return plots_created

    def create_benchmark_visualizations(self, results: Dict[str, Any]) -> List[Path]:
        """Create visualizations for benchmark results."""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available, skipping charts")
            return []

        plots_created = []

        # Performance and Latency Charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Performance metrics
        if "performance" in results:
            perf = results["performance"]
            metrics = [
                "Throughput\n(RPS)",
                "Avg Response\nTime (s)",
                "P95 Response\nTime (s)",
                "Error Rate\n(%)",
            ]
            values = [
                perf["throughput_rps"],
                perf["avg_response_time"],
                perf["p95_response_time"],
                perf["error_rate"] * 100,
            ]

            bars = axes[0, 0].bar(
                metrics, values, color=["lightblue", "lightgreen", "orange", "lightcoral"]
            )
            axes[0, 0].set_title("Performance Metrics")
            axes[0, 0].tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                )

        # Latency percentiles
        if "latency" in results:
            latency = results["latency"]
            percentiles = ["P50", "P75", "P90", "P95", "P99", "P99.9"]
            values = [latency["percentiles"][f"p{p}"] for p in [50, 75, 90, 95, 99, 99.9]]

            axes[0, 1].plot(percentiles, values, "o-", linewidth=2, markersize=8)
            axes[0, 1].set_title("Response Time Percentiles")
            axes[0, 1].set_ylabel("Response Time (s)")
            axes[0, 1].grid(True, alpha=0.3)

        # Stress test results
        if "stress" in results:
            stress = results["stress"]

            # Throughput comparison
            throughput_labels = ["Average", "Peak"]
            throughput_values = [stress["avg_throughput_rps"], stress["peak_throughput_rps"]]

            bars = axes[1, 0].bar(
                throughput_labels, throughput_values, color=["lightblue", "darkblue"]
            )
            axes[1, 0].set_title("Stress Test Throughput")
            axes[1, 0].set_ylabel("Throughput (RPS)")

            for bar, value in zip(bars, throughput_values):
                height = bar.get_height()
                axes[1, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                )

            # Success rate and stability
            metrics = ["Success Rate", "Stability Score"]
            values = [
                stress["successful_requests"] / stress["total_requests"],
                stress["stability_score"],
            ]
            colors = ["green" if v > 0.9 else "orange" if v > 0.7 else "red" for v in values]

            bars = axes[1, 1].bar(metrics, values, color=colors)
            axes[1, 1].set_title("Reliability Metrics")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].set_ylim(0, 1)

            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        benchmark_path = self.output_dir / "benchmark_charts.png"
        plt.savefig(benchmark_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots_created.append(benchmark_path)

        return plots_created

    def generate_html_report(
        self, results: Dict[str, Any], charts: List[Path], report_type: str = "quality"
    ) -> Path:
        """Generate an HTML report with embedded charts."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Industrial Audio RAG - {report_type.title()} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ background: #ecf0f1; padding: 10px; margin: 10px 0;
                    border-radius: 5px; }}
                .success {{ color: #27ae60; font-weight: bold; }}
                .failure {{ color: #e74c3c; font-weight: bold; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Industrial Audio RAG - {report_type.title()} Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """

        if report_type == "quality":
            # Quality report content
            overall_score = results.get("overall_score", 0)
            passed = results.get("passed_quality_gate", False)

            html_content += f"""
            <h2>Executive Summary</h2>
            <div class="metric">
                <strong>Overall Score:</strong> {overall_score:.3f}<br>
                <strong>Quality Gate:</strong> <span class="{"success" if passed else "failure"}">
                    {"✅ PASSED" if passed else "❌ FAILED"}</span><br>
                <strong>Success Rate:</strong>
                {results.get("passed_queries", 0)}/{results.get("total_queries", 0)}
            </div>
            """

        elif report_type == "benchmark":
            # Benchmark report content
            if "performance" in results:
                perf = results["performance"]
                html_content += f"""
                <h2>Performance Summary</h2>
                <div class="metric">
                    <strong>Throughput:</strong> {perf["throughput_rps"]:.2f} RPS<br>
                    <strong>Avg Response Time:</strong> {perf["avg_response_time"]:.3f}s<br>
                    <strong>P95 Response Time:</strong> {perf["p95_response_time"]:.3f}s<br>
                    <strong>Error Rate:</strong> {perf["error_rate"]:.1%}
                </div>
                """

        # Add charts
        if charts:
            html_content += "<h2>Charts and Visualizations</h2>"
            for chart_path in charts:
                chart_name = chart_path.stem.replace("_", " ").title()
                # Convert to relative path for HTML
                rel_path = chart_path.name
                html_content += f"""
                <div class="chart">
                    <h3>{chart_name}</h3>
                    <img src="{rel_path}" alt="{chart_name}">
                </div>
                """

        html_content += """
        </body>
        </html>
        """

        html_path = self.output_dir / f"{report_type}_report.html"
        with open(html_path, "w") as f:
            f.write(html_content)

        return html_path

    def generate_comprehensive_report(
        self, results_path: Path, report_type: str = "quality"
    ) -> Dict[str, Path]:
        """Generate a comprehensive report with text, charts, and HTML."""
        logger.info(f"Generating {report_type} report from {results_path}")

        results = self.load_results(results_path)
        generated_files = {}

        # Generate text report
        if report_type == "quality":
            text_report = self.generate_quality_report(results)
            charts = self.create_quality_visualizations(results)
        elif report_type == "benchmark":
            text_report = self.generate_benchmark_report(results)
            charts = self.create_benchmark_visualizations(results)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

        # Save text report
        text_path = self.output_dir / f"{report_type}_report.md"
        with open(text_path, "w") as f:
            f.write(text_report)
        generated_files["text_report"] = text_path

        # Generate HTML report
        html_path = self.generate_html_report(results, charts, report_type)
        generated_files["html_report"] = html_path

        # Add chart paths
        if charts:
            generated_files["charts"] = charts

        logger.info("Report generated successfully:")
        for file_type, file_path in generated_files.items():
            if file_type == "charts":
                logger.info(f"  Charts: {len(file_path)} files in {self.output_dir}")
            else:
                logger.info(f"  {file_type}: {file_path}")

        return generated_files


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Generate evaluation reports and visualizations")
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval/reports"), help="Output directory for reports"
    )
    parser.add_argument(
        "--type",
        choices=["quality", "benchmark"],
        default="quality",
        help="Type of report to generate",
    )
    parser.add_argument(
        "--format", choices=["text", "html", "all"], default="all", help="Report format to generate"
    )

    args = parser.parse_args()

    try:
        # Initialize reporter
        reporter = EvaluationReporter(args.output)

        # Generate comprehensive report
        generated_files = reporter.generate_comprehensive_report(args.input, args.type)

        print(f"\n{'=' * 60}")
        print("REPORT GENERATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Report Type: {args.type.title()}")
        print(f"Output Directory: {args.output}")
        print("\nGenerated Files:")

        for file_type, file_path in generated_files.items():
            if file_type == "charts":
                print(f"  📊 Charts: {len(file_path)} visualization files")
            else:
                print(f"  📄 {file_type.replace('_', ' ').title()}: {file_path.name}")

        # Show next steps
        html_report = generated_files.get("html_report")
        if html_report:
            print("\n🌐 Open the HTML report in your browser:")
            print(f"   file://{html_report.absolute()}")

        return 0

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
