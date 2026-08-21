#!/usr/bin/env python3
"""Profile where time is spent in the alignment path."""

import time
from cite_right import align_citations
from cite_right.core.prepared_corpus import PreparedCitationCorpus
from cite_right.core.citation_config import CitationConfig

# Test data
SOURCES = [
    f"""Document {i}: Solar panel efficiency has improved dramatically.
    Modern photovoltaic cells achieve over 22% efficiency.
    Manufacturing costs have dropped by 90% since 2010."""
    for i in range(50)
]

SOURCES[25] = """The transition to renewable energy accelerates globally.
Solar and wind power now represent cheap electricity sources.
Battery storage technologies enable grid stability."""

ANSWER = """The transition to renewable energy accelerates globally.
Solar and wind power represent electricity sources."""

def profile_alignment():
    """Profile the alignment path."""
    
    # Build corpus
    t0 = time.perf_counter()
    corpus = PreparedCitationCorpus.from_sources(SOURCES, use_rust=True)
    prepare_time = (time.perf_counter() - t0) * 1000
    
    # Run alignment with instrumentation
    t0 = time.perf_counter()
    
    # Monkey-patch to add timing
    from cite_right.core import aligner_rust
    original_align_batch = aligner_rust.RustSmithWatermanAligner.align_batch
    
    align_batch_time = [0.0]
    align_batch_calls = [0]
    
    def timed_align_batch(self, seq1, seqs):
        t = time.perf_counter()
        result = original_align_batch(self, seq1, seqs)
        align_batch_time[0] += (time.perf_counter() - t) * 1000
        align_batch_calls[0] += 1
        return result
    
    aligner_rust.RustSmithWatermanAligner.align_batch = timed_align_batch
    
    result = corpus.align(ANSWER, backend='rust')
    total_align_time = (time.perf_counter() - t0) * 1000
    
    # Restore
    aligner_rust.RustSmithWatermanAligner.align_batch = original_align_batch
    
    print(f"Prepare time: {prepare_time:.2f}ms")
    print(f"Total align time: {total_align_time:.2f}ms")
    print(f"  SW align_batch calls: {align_batch_calls[0]}")
    print(f"  SW align_batch time: {align_batch_time[0]:.2f}ms ({align_batch_time[0]/total_align_time*100:.0f}%)")
    print(f"  Python overhead: {total_align_time - align_batch_time[0]:.2f}ms ({(total_align_time - align_batch_time[0])/total_align_time*100:.0f}%)")
    print(f"\nTotal: {prepare_time + total_align_time:.2f}ms")

if __name__ == "__main__":
    # Warmup
    for _ in range(3):
        PreparedCitationCorpus.from_sources(SOURCES[:2], use_rust=True)
    
    # Profile
    for i in range(5):
        print(f"\n=== Run {i+1} ===")
        profile_alignment()
