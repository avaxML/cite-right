#!/usr/bin/env python3
"""Profiled benchmark."""

import time
import statistics
from cite_right.core.prepared_corpus import PreparedCitationCorpus
from cite_right.core import aligner_rust

# Same workload as benchmark_rust_prepare.py
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
] * 5  # 50 sources

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

# Monkey-patch timing
original_align_batch = aligner_rust.RustSmithWatermanAligner.align_batch
align_batch_time = []
align_batch_calls = []

def timed_align_batch(self, seq1, seqs):
    t = time.perf_counter()
    result = original_align_batch(self, seq1, seqs)
    elapsed = (time.perf_counter() - t) * 1000
    align_batch_time.append(elapsed)
    align_batch_calls.append(1)
    return result

aligner_rust.RustSmithWatermanAligner.align_batch = timed_align_batch

# Warmup
print("Warming up...")
corpus = PreparedCitationCorpus.from_sources(SOURCES[:2], use_rust=True)
corpus.align(ANSWER[:100], backend='rust')
align_batch_time.clear()
align_batch_calls.clear()

# Benchmark
print("Benchmarking...")
corpus = PreparedCitationCorpus.from_sources(SOURCES, use_rust=True)
align_times = []
for _ in range(30):
    batch_times_before = len(align_batch_time)
    t = time.perf_counter()
    corpus.align(ANSWER, backend='rust')
    total_time = (time.perf_counter() - t) * 1000
    batch_times_after = len(align_batch_time)
    sw_time_this_run = sum(align_batch_time[batch_times_before:batch_times_after])
    align_times.append((total_time, sw_time_this_run))

aligner_rust.RustSmithWatermanAligner.align_batch = original_align_batch

# Stats
total_p50 = statistics.median([t for t, _ in align_times])
sw_p50 = statistics.median([s for _, s in align_times])
python_p50 = total_p50 - sw_p50

print(f"\nAlign p50: {total_p50:.2f}ms")
print(f"  SW batch: {sw_p50:.2f}ms ({sw_p50/total_p50*100:.0f}%)")
print(f"  Python overhead: {python_p50:.2f}ms ({python_p50/total_p50*100:.0f}%)")
print(f"  SW calls per align: {len(align_batch_calls) / len(align_times):.1f}")
