#!/usr/bin/env python3
"""Measure oneshot p50/p95 vs main branch baseline."""

import statistics
import time

from cite_right import align_citations
from cite_right.core.prepared_corpus import PreparedCitationCorpus

# Large realistic RAG workload
ANSWER = """
The transition to renewable energy accelerates globally as costs decline.
Solar and wind power now represent the cheapest sources of electricity in most markets.
Battery storage technologies enable grid stability with intermittent renewables.
Electric vehicles are rapidly displacing internal combustion engines worldwide.
"""

# 100 source documents
SOURCES = [
    f"""Document {i}: Solar panel efficiency has improved dramatically over the past decade.
    Modern photovoltaic cells achieve over 22% conversion efficiency in commercial applications.
    Manufacturing costs have dropped by 90% since 2010, making solar competitive with fossil fuels.
    Large-scale solar farms now generate electricity at under 3 cents per kilowatt-hour.
    Energy storage integration addresses intermittency challenges in renewable power systems."""
    for i in range(100)
]

# Insert relevant content in middle
SOURCES[50] = """
The transition to renewable energy accelerates globally as investment flows increase.
Solar and wind power installations broke records in 2025 across all major markets worldwide.
Battery storage technologies like lithium-ion and flow batteries enable reliable grid integration.
Electric vehicles reached 25% market share in Europe, displacing traditional internal combustion engines.
Policy support and falling costs drive the clean energy transformation at unprecedented pace.
"""


def benchmark_oneshot(use_rust: bool, iterations: int = 20) -> dict:
    """Benchmark full oneshot align_citations."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        _result = align_citations(ANSWER, SOURCES)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return {
        "p50": statistics.median(times),
        "p95": statistics.quantiles(times, n=20)[18]
        if len(times) >= 20
        else max(times),
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
    }


def benchmark_prepared(use_rust: bool, iterations: int = 30) -> dict:
    """Benchmark prepare-once reuse-many pattern."""
    corpus = PreparedCitationCorpus.from_sources(SOURCES, use_rust=use_rust)
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        _result = corpus.align(ANSWER, backend="rust")
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return {
        "p50": statistics.median(times),
        "p95": statistics.quantiles(times, n=20)[18]
        if len(times) >= 20
        else max(times),
        "mean": statistics.mean(times),
    }


def main() -> None:
    print("=" * 80)
    print("REALISTIC RAG WORKLOAD BENCHMARK")
    print("=" * 80)
    print(
        f"\nWorkload: {len(SOURCES)} sources, ~{sum(len(s) for s in SOURCES):,} chars total"
    )
    print(f"Answer: {len(ANSWER)} chars, {len(ANSWER.split())} words")

    # Warmup
    print("\nWarming up...")
    PreparedCitationCorpus.from_sources(SOURCES[:5], use_rust=False)
    PreparedCitationCorpus.from_sources(SOURCES[:5], use_rust=True)

    # Oneshot benchmark
    print("\n" + "-" * 80)
    print("ONESHOT (cold start, build corpus + align)")
    print("-" * 80)

    print("\nPython baseline:")
    # Disable Rust temporarily
    import cite_right.core.prepared_corpus as pc

    original = pc.RUST_PREPARE_AVAILABLE
    pc.RUST_PREPARE_AVAILABLE = False
    python_oneshot = benchmark_oneshot(False, iterations=10)
    pc.RUST_PREPARE_AVAILABLE = original
    print(f"  p50: {python_oneshot['p50']:.1f} ms")
    print(f"  p95: {python_oneshot['p95']:.1f} ms")

    print("\nRust prepare:")
    rust_oneshot = benchmark_oneshot(True, iterations=10)
    print(f"  p50: {rust_oneshot['p50']:.1f} ms")
    print(f"  p95: {rust_oneshot['p95']:.1f} ms")

    print(
        f"\nOneshot speedup: {python_oneshot['p50'] / rust_oneshot['p50']:.1f}x (p50), {python_oneshot['p95'] / rust_oneshot['p95']:.1f}x (p95)"
    )

    # Prepared corpus benchmark
    print("\n" + "-" * 80)
    print("PREPARED CORPUS (reuse pattern)")
    print("-" * 80)

    print("\nPython prepare:")
    pc.RUST_PREPARE_AVAILABLE = False
    python_prepared = benchmark_prepared(False, iterations=20)
    pc.RUST_PREPARE_AVAILABLE = original
    print(f"  p50: {python_prepared['p50']:.1f} ms")
    print(f"  p95: {python_prepared['p95']:.1f} ms")

    print("\nRust prepare:")
    rust_prepared = benchmark_prepared(True, iterations=20)
    print(f"  p50: {rust_prepared['p50']:.1f} ms")
    print(f"  p95: {rust_prepared['p95']:.1f} ms")

    print(
        f"\nPrepared speedup: {python_prepared['p50'] / rust_prepared['p50']:.1f}x (p50), {python_prepared['p95'] / rust_prepared['p95']:.1f}x (p95)"
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"Oneshot: {python_oneshot['p50']:.1f}ms → {rust_oneshot['p50']:.1f}ms ({python_oneshot['p50'] / rust_oneshot['p50']:.1f}x)"
    )
    print(
        f"Prepared: {python_prepared['p50']:.1f}ms → {rust_prepared['p50']:.1f}ms ({python_prepared['p50'] / rust_prepared['p50']:.1f}x)"
    )


if __name__ == "__main__":
    main()
