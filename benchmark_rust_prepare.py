#!/usr/bin/env python3
"""Benchmark Rust prepare path vs Python baseline."""

import statistics
import time

from cite_right.core.prepared_corpus import PreparedCitationCorpus

# RAG-style test data - scaled up to realistic size
ANSWER = """
Cloud computing has fundamentally transformed how modern businesses operate in the digital age.
Companies can now dynamically scale their infrastructure resources on demand without requiring 
massive upfront capital investments in physical hardware. The benefits are numerous and include 
operational flexibility, significant cost savings through pay-as-you-go models, global reach 
with distributed data centers, and the ability to innovate faster without infrastructure constraints.
However, security and compliance concerns remain top priorities for enterprise adoption, particularly 
in regulated industries like healthcare and finance. Organizations must carefully evaluate their 
cloud strategy, considering factors such as data residency requirements, vendor lock-in risks, and 
the trade-offs between public, private, and hybrid cloud architectures.
"""

# Simulate realistic RAG corpus: 50 sources, each with substantial content
SOURCES = [
    """Cloud computing fundamentally changed enterprise IT infrastructure by providing on-demand 
    access to computing resources over the internet. Traditional data centers required significant 
    capital expenditure, long procurement cycles, and extensive physical maintenance. Modern cloud 
    platforms eliminate these barriers by offering elastic scalability, usage-based pricing, and 
    managed services that abstract away infrastructure complexity. Organizations can provision virtual 
    machines, storage, databases, and networking resources within minutes rather than months.""",
    """Organizations worldwide leverage cloud services to dramatically reduce capital expenditures 
    on hardware and data center facilities. The shift from CapEx to OpEx models allows businesses 
    to convert fixed costs into variable costs that scale with actual usage. This financial flexibility 
    is particularly valuable for startups and growing companies that need to manage cash flow carefully. 
    Cloud economics enable experimentation without large upfront commitments, fostering innovation and 
    agility in competitive markets.""",
    """The scalability of cloud infrastructure allows businesses to grow without physical constraints 
    or capacity planning headaches. Auto-scaling groups automatically add or remove compute capacity 
    based on demand patterns, ensuring applications remain responsive during traffic spikes while 
    minimizing costs during quiet periods. This elasticity is impossible to achieve economically with 
    on-premises infrastructure, where capacity must be provisioned for peak load rather than average 
    utilization.""",
    """Major cloud providers including AWS, Microsoft Azure, and Google Cloud offer data centers 
    distributed globally across multiple geographic regions and availability zones. This global 
    infrastructure enables businesses to deploy applications close to end users, reducing latency 
    and improving user experience. Geographic distribution also supports disaster recovery strategies, 
    regulatory compliance with data residency requirements, and high availability architectures that 
    can withstand regional outages.""",
    """Security in cloud environments requires careful configuration, continuous monitoring, and 
    adherence to the shared responsibility model. While cloud providers secure the underlying 
    infrastructure, customers remain responsible for securing their data, applications, and access 
    controls. Common security challenges include misconfigured storage buckets, overly permissive 
    IAM policies, and insufficient logging. Organizations must implement defense-in-depth strategies, 
    including encryption at rest and in transit, network segmentation, and regular security audits.""",
    """Enterprise cloud adoption faces significant challenges including data sovereignty regulations, 
    compliance requirements, and concerns about vendor lock-in. Industries like healthcare and finance 
    must navigate complex regulatory frameworks such as HIPAA, GDPR, and PCI-DSS when moving workloads 
    to the cloud. Data residency laws in various jurisdictions may require data to remain within specific 
    geographic boundaries, complicating cloud architecture decisions and potentially limiting the benefits 
    of global infrastructure.""",
    """Cost optimization in the cloud involves rightsizing resources, leveraging reserved instances 
    or savings plans, and implementing automated policies to eliminate waste. Many organizations 
    overprovision resources out of caution, paying for capacity they don't need. Cloud cost management 
    tools help identify idle resources, oversized instances, and opportunities to use spot instances 
    for non-critical workloads. Effective cost optimization requires ongoing monitoring, clear tagging 
    strategies, and organizational discipline.""",
    """Multi-cloud strategies help organizations avoid vendor lock-in, improve resilience, and 
    negotiate better pricing through competitive leverage. However, multi-cloud architectures 
    introduce significant complexity in areas like networking, identity management, and observability. 
    Organizations must balance the benefits of flexibility and redundancy against the operational 
    overhead of managing multiple cloud platforms with different APIs, tools, and pricing models. 
    Many companies find that multi-cloud reality is messier than the promise.""",
    """Serverless computing abstracts infrastructure management entirely, allowing developers to 
    focus on writing code rather than managing servers. Functions-as-a-Service (FaaS) platforms 
    like AWS Lambda automatically handle scaling, patching, and availability. Serverless architectures 
    excel for event-driven workloads, API backends, and data processing pipelines. However, they 
    introduce new challenges around cold starts, execution time limits, and debugging distributed 
    systems composed of many small functions.""",
    """Container orchestration platforms like Kubernetes have become the de facto standard for 
    deploying and managing containerized applications in cloud environments. Kubernetes provides 
    powerful primitives for service discovery, load balancing, rolling updates, and self-healing. 
    However, Kubernetes complexity is significant, requiring specialized expertise to operate 
    production clusters securely and reliably. Managed Kubernetes services from cloud providers 
    reduce but don't eliminate this operational burden.""",
] * 5  # Repeat to get 50 sources


def benchmark_prepare_path(use_rust: bool, iterations: int = 10) -> dict:
    """Benchmark corpus preparation."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        _corpus = PreparedCitationCorpus.from_sources(SOURCES, use_rust=use_rust)
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
        "samples": len(times),
    }


def benchmark_end_to_end(use_rust: bool, iterations: int = 10) -> dict:
    """Benchmark full align_citations."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        # Build corpus with prepare phase
        corpus = PreparedCitationCorpus.from_sources(SOURCES, use_rust=use_rust)
        # Run alignment (SW in Python for both)
        _result = corpus.align(ANSWER, backend="rust")
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
        "samples": len(times),
    }


def main() -> None:
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

    speedup_p50 = python_prepare["p50"] / rust_prepare["p50"]
    speedup_p95 = python_prepare["p95"] / rust_prepare["p95"]
    print("\nSpeedup (Rust vs Python):")
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

    e2e_speedup_p50 = python_e2e["p50"] / rust_e2e["p50"]
    e2e_speedup_p95 = python_e2e["p95"] / rust_e2e["p95"]
    print("\nEnd-to-end speedup:")
    print(f"  p50: {e2e_speedup_p50:.1f}x")
    print(f"  p95: {e2e_speedup_p95:.1f}x")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Prepare path speedup: {speedup_p50:.1f}x (p50), {speedup_p95:.1f}x (p95)")
    print(
        f"End-to-end speedup: {e2e_speedup_p50:.1f}x (p50), {e2e_speedup_p95:.1f}x (p95)"
    )
    print("\nBottleneck analysis:")
    prepare_fraction = rust_prepare["p50"] / rust_e2e["p50"] * 100
    sw_fraction = 100 - prepare_fraction
    print(f"  Prepare path: ~{prepare_fraction:.0f}% of end-to-end time")
    print(f"  SW alignment: ~{sw_fraction:.0f}% of end-to-end time")


if __name__ == "__main__":
    main()
