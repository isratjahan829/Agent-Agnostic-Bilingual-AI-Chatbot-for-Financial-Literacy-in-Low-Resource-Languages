import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from banglafingpt.config import load_config
from banglafingpt.data.schema import QAPair, Segment
from banglafingpt.pipeline import BanglaFinGPT
from banglafingpt.retrieval.retriever import HybridRetriever
from banglafingpt.utils import read_jsonl


@pytest.fixture(scope="session")
def sample_segments():
    return [
        Segment(**{k: v for k, v in row.items() if k in Segment.__dataclass_fields__})
        for row in read_jsonl(ROOT / "data/sample/segments.jsonl")
    ]


@pytest.fixture(scope="session")
def sample_pairs():
    return [QAPair.from_dict(row) for row in read_jsonl(ROOT / "data/sample/qa_pairs.jsonl")]


@pytest.fixture(scope="session")
def demo_config():
    return load_config(ROOT / "configs/demo.yaml")


@pytest.fixture(scope="session")
def retriever(sample_segments, demo_config):
    return HybridRetriever.from_segments(sample_segments, demo_config.retrieval)


@pytest.fixture()
def system(demo_config, retriever):
    return BanglaFinGPT(demo_config, retriever=retriever)
