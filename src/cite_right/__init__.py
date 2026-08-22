from cite_right.citations import (
    AlignmentMetrics,
    PreparedCitationCorpus,
    align_citations,
)
from cite_right.claims import (
    Claim,
    ClaimDecomposer,
    SimpleClaimDecomposer,
    SpacyClaimDecomposer,
)
from cite_right.convenience import (
    annotate_answer,
    check_groundedness,
    format_with_citations,
    get_citation_summary,
    is_grounded,
    is_hallucinated,
)
from cite_right.core.citation_config import CitationConfig, CitationWeights
from cite_right.core.results import (
    AnswerSpan,
    Citation,
    EvidenceSpan,
    RetrievalSupport,
    Segment,
    SourceChunk,
    SourceDocument,
    SpanCitations,
    TokenizedText,
)
from cite_right.fact_verification import (
    ClaimVerification,
    FactVerificationConfig,
    FactVerificationMetrics,
    verify_facts,
)
from cite_right.hallucination import (
    HallucinationConfig,
    HallucinationMetrics,
    SpanConfidence,
    compute_hallucination_metrics,
)
from cite_right.integrations import (
    LANGCHAIN_AVAILABLE,
    LLAMAINDEX_AVAILABLE,
    LangChainDocument,
    LlamaIndexNode,
    LlamaIndexNodeWithScore,
    LlamaIndexTextNode,
    from_dicts,
    from_langchain_chunks,
    from_langchain_documents,
    from_llamaindex_chunks,
    from_llamaindex_nodes,
    is_langchain_available,
    is_langchain_document,
    is_llamaindex_available,
    is_llamaindex_node,
)
from cite_right.text.answer_segmenter_spacy import SpacyAnswerSegmenter

# Optional embedders - import only if dependencies are available
try:
    from cite_right.models.sbert_embedder import SentenceTransformerEmbedder
except ImportError:
    SentenceTransformerEmbedder = None  # type: ignore

try:
    from cite_right.models.onnx_embedder import OnnxMiniLmEmbedder
except ImportError:
    OnnxMiniLmEmbedder = None  # type: ignore
from cite_right.text.segmenter_pysbd import PySBDSegmenter
from cite_right.text.segmenter_spacy import SpacySegmenter
from cite_right.text.tokenizer import SimpleTokenizer, TokenizerConfig
from cite_right.text.tokenizer_huggingface import HuggingFaceTokenizer
from cite_right.text.tokenizer_tiktoken import TiktokenTokenizer

__version__ = "0.3.1"

__all__ = [
    "__version__",
    # Core API
    "align_citations",
    "compute_hallucination_metrics",
    "verify_facts",
    # Convenience functions
    "annotate_answer",
    "check_groundedness",
    "format_with_citations",
    "get_citation_summary",
    "is_grounded",
    "is_hallucinated",
    # Framework integrations
    "LANGCHAIN_AVAILABLE",
    "LLAMAINDEX_AVAILABLE",
    "LangChainDocument",
    "LlamaIndexNode",
    "LlamaIndexNodeWithScore",
    "LlamaIndexTextNode",
    "from_dicts",
    "from_langchain_chunks",
    "from_langchain_documents",
    "from_llamaindex_chunks",
    "from_llamaindex_nodes",
    "is_langchain_available",
    "is_langchain_document",
    "is_llamaindex_available",
    "is_llamaindex_node",
    # Configuration
    "CitationConfig",
    "CitationWeights",
    "FactVerificationConfig",
    "HallucinationConfig",
    "TokenizerConfig",
    # Result types
    "AlignmentMetrics",
    "AnswerSpan",
    "Citation",
    "PreparedCitationCorpus",
    "Claim",
    "ClaimVerification",
    "EvidenceSpan",
    "FactVerificationMetrics",
    "HallucinationMetrics",
    "RetrievalSupport",
    "Segment",
    "SourceChunk",
    "SourceDocument",
    "SpanCitations",
    "SpanConfidence",
    "TokenizedText",
    # Tokenizers
    "HuggingFaceTokenizer",
    "SimpleTokenizer",
    "TiktokenTokenizer",
    # Segmenters
    "PySBDSegmenter",
    "SpacyAnswerSegmenter",
    "SpacySegmenter",
    # Claim decomposition
    "ClaimDecomposer",
    "SimpleClaimDecomposer",
    "SpacyClaimDecomposer",
    # Embedders
    "SentenceTransformerEmbedder",
    "OnnxMiniLmEmbedder",
]
