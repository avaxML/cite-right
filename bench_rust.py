#!/usr/bin/env python3
"""Benchmark Rust prepare path vs Python baseline."""

import time
import statistics
from cite_right.core.prepared_corpus import PreparedCitationCorpus

# RAG-style test data
ANSWER = """
Cloud computing has transformed how businesses operate. Companies can now scale 
their infrastructure on demand without massive upfront investments. The benefits 
include flexibility, cost savings, and global reach. However, security concerns 
remain a top priority for enterprise adoption.
"""

SOURCES = [
    "Cloud computing provides on-demand access to computing resources over the internet.",
    "Organizations leverage cloud services to reduce capital expenditures on hardware.",
    "The scalability of cloud infrastructure allows businesses to grow without physical constraints.",
    "Major cloud providers offer data centers distributed globally for low latency.",
    "Security in cloud environments requires careful configuration and monitoring.",
    "Enterprise cloud adoption faces challenges including data sovereignty and compliance.",
    "Cost optimization in the cloud involves rightsizing resources and using reserved instances.",
    "Multi-cloud strategies help organizations avoid vendor lock-in.",
    "Serverless computing abstracts infrastructure management entirely.",
    "Container orchestration platforms like Kubernetes are popular in cloud deployments.",
]

def benchmark_prepare_path(use_rust: bool, iterations: int = 10) -> dict:
    """Benchmark corpus preparation."""
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        corpus = PreparedCitationCorpus.from_sources(SOURCES, use_rust=use_rust)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
    
    return {
        "p50": statistics.median(times),
        "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "samples": len(times),
    }

def benchmark_end_to_end(use_rust: bool, iterations: int = 10) -> dict:
    """Benchmark full pipeline."""
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        corpus = PreparedCitationCorpus.from_sources(SOURCES, use_rust=use_rust)
        result = corpus.align(ANSWER, backend="rust")
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)
    
    return {
        "p50": statistics.median(times),
        "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "samples": len(times),
    }

def main():
    print("=" * 80)
    print("RUST PREPARE PATH BENCHMARK")
    print("=" * 80)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        PreparedCitationCorpus.from_sources(SOURCES[:2], use_rust=False)
        PreparedCitationCorpus.from_sources(SOURCES[:2], use_rust=True)
    
    # Benchmark prepare path only
    print("\n" + "-" * 80)
    print("PREPARE PATH (tokenize + passages + candidates + IDF)")
    print("-" * 80)
    
    print("\nPython baseline (use_rust=False):")
    python_prepare = benchmark_prepare_path(use_rust=False, iterations=50)
    print(f"  p50: {python_prepare['p50']:.2f} ms")
    print(f"  p95: {python_prepare['p95']:.2f} ms")
    print(f"  mean: {python_prepare['mean']:.2f} ms")
    
    print("\nRust path (use_rust=True):")
    rust_prepare = benchmark_prepare_path(use_rust=True, iterations=50)
    print(f"  p50: {rust_prepare['p50']:.2f} ms")
    print(f"  p95: {rust_prepare['p95']:.2f} ms")
    print(f"  mean: {rust_prepare['mean']:.2f} ms")
    
    speedup_p50 = python_prepare['p50'] / rust_prepare['p50']
    speedup_p95 = python_prepare['p95'] / rust_prepare['p95']
    print(f"\nSpeedup (Rust vs Python):")
    print(f"  p50: {speedup_p50:.1f}x")
    print(f"  p95: {speedup_p95:.1f}x")
    
    # End-to-end benchmark
    print("\n" + "-" * 80)
    print("END-TO-END (prepare + align with Smith-Waterman)")
    print("-" * 80)
    
    print("\nPython baseline:")
    python_e2e = benchmark_end_to_end(use_rust=False, iterations=30)
    print(f"  p50: {python_e2e['p50']:.2f} ms")
    print(f"  p95: {python_e2e['p95']:.2f} ms")
    
    print("\nRust prepare + Python SW:")
    rust_e2e = benchmark_end_to_end(use_rust=True, iterations=30)
    print(f"  p50: {rust_e2e['p50']:.2f} ms")
    print(f"  p95: {rust_e2e['p95']:.2f} ms")
    
    e2e_speedup_p50 = python_e2e['p50'] / rust_e2e['p50']
    e2e_speedup_p95 = python_e2e['p95'] / rust_e2e['p95']
    print(f"\nEnd-to-end speedup:")
    print(f"  p50: {e2e_speedup_p50:.1f}x")
    print(f"  p95: {e2e_speedup_p95:.1f}x")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Prepare path speedup: {speedup_p50:.1f}x (p50), {speedup_p95:.1f}x (p95)")
    print(f"End-to-end speedup: {e2e_speedup_p50:.1f}x (p50), {e2e_speedup_p95:.1f}x (p95)")
    print("\nBottleneck analysis:")
    prepare_fraction = rust_prepare['p50'] / rust_e2e['p50'] * 100
    sw_fraction = 100 - prepare_fraction
    print(f"  Prepare path: ~{prepare_fraction:.0f}% of end-to-end time")
    print(f"  SW alignment: ~{sw_fraction:.0f}% of end-to-end time")

if __name__ == "__main__":
    main()
