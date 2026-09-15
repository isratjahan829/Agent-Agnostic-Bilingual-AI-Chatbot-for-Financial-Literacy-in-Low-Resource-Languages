from banglafingpt.retrieval.embedder import HashingEmbedder
from banglafingpt.retrieval.index import BM25, DocumentIndex
from banglafingpt.retrieval.retriever import chunks_from_segments
from banglafingpt.utils import content_tokens


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=64)
    a, b = embedder.encode(["ভ্যাট হার"])[0], embedder.encode(["ভ্যাট হার"])[0]
    assert a == b
    assert abs(sum(v * v for v in a) ** 0.5 - 1.0) < 1e-6


def test_bm25_ranks_the_matching_document_first():
    corpus = [content_tokens(t) for t in ["ভ্যাটের হার ১৫ শতাংশ", "আয়কর রিটার্ন নভেম্বর"]]
    scores = BM25(corpus).scores(content_tokens("ভ্যাটের হার"))
    assert scores[0] > scores[1]


def test_retriever_finds_the_right_segment(retriever):
    results = retriever.retrieve("ভ্যাটের আদর্শ হার কত?", top_k=3)
    assert results
    assert "১৫ শতাংশ" in results[0].text


def test_domain_filter_restricts_results(retriever):
    results = retriever.retrieve("হার কত?", top_k=5, domain="customs")
    assert results and all(c.domain == "customs" for c in results)


def test_index_roundtrips_through_disk(tmp_path, sample_segments):
    index = DocumentIndex(HashingEmbedder(dim=64)).build(chunks_from_segments(sample_segments))
    index.save(tmp_path / "idx")
    loaded = DocumentIndex.load(tmp_path / "idx", embedder=HashingEmbedder(dim=64))
    assert len(loaded) == len(index)
    assert loaded.chunks[0].text == index.chunks[0].text
