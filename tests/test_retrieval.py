from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from taxi_agent.retrieval import SchemaRetriever, SchemaRetrieverConfig
from taxi_agent.schema import ColumnSchema, TableSchema


class BrokenEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise RuntimeError("embedding provider unavailable")

    def embed_query(self, text: str) -> List[float]:
        raise RuntimeError("embedding provider unavailable")


class FlakyEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        _ = texts
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary embedding failure")
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        _ = text
        return [0.1, 0.2, 0.3]


class StableEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        _ = text
        return [0.1, 0.2, 0.3]


def _sample_tables() -> List[TableSchema]:
    return [
        TableSchema(
            table_schema="public",
            table_name="taxi_trip_data",
            columns=[
                ColumnSchema("payment_type", "integer", 1),
                ColumnSchema("fare_amount", "double precision", 2),
                ColumnSchema("tip_amount", "double precision", 3),
            ],
        ),
        TableSchema(
            table_schema="public",
            table_name="zones",
            columns=[
                ColumnSchema("zone_id", "integer", 1),
                ColumnSchema("borough", "text", 2),
            ],
        ),
    ]


def test_retrieval_with_bm25_only() -> None:
    retriever = SchemaRetriever(
        embedding_model=None,
        config=SchemaRetrieverConfig(top_k_tables=1, search_type="mmr", fetch_k=20),
    )
    retriever.refresh(_sample_tables())
    selected = retriever.retrieve_tables("average fare and tip by payment type")
    assert len(selected) == 1
    assert selected[0].table_name in {"taxi_trip_data", "zones"}


def test_embedding_failure_falls_back_to_keyword() -> None:
    retriever = SchemaRetriever(
        embedding_model=BrokenEmbeddings(),
        config=SchemaRetrieverConfig(top_k_tables=1, search_type="mmr", fetch_k=20),
    )
    retriever.refresh(_sample_tables())
    selected = retriever.retrieve_tables("borough list")
    assert len(selected) == 1
    assert selected[0].table_name in {"zones", "taxi_trip_data"}


def test_bm25_retried_when_previous_build_failed(monkeypatch) -> None:
    class FlakyBM25:
        calls = 0

        @classmethod
        def from_documents(cls, docs):
            cls.calls += 1
            if cls.calls == 1:
                raise RuntimeError("temporary bm25 failure")
            inst = cls()
            inst.docs = docs
            inst.k = 1
            return inst

        def invoke(self, question: str):
            _ = question
            return [
                Document(
                    page_content="Table: public.taxi_trip_data",
                    metadata={"full_name": "public.taxi_trip_data"},
                )
            ]

    monkeypatch.setattr("taxi_agent.retrieval.BM25Retriever", FlakyBM25)

    retriever = SchemaRetriever(
        embedding_model=None,
        config=SchemaRetrieverConfig(top_k_tables=1, search_type="mmr", fetch_k=20),
    )
    tables = _sample_tables()

    retriever.refresh(tables)
    assert retriever.keyword_retriever is None

    # Same schema fingerprint; refresh should retry BM25 initialization.
    retriever.refresh(tables)
    assert retriever.keyword_retriever is not None


def test_bm25_missing_dependency_not_retried_for_same_fingerprint(monkeypatch) -> None:
    class MissingBM25:
        calls = 0

        @classmethod
        def from_documents(cls, docs):
            _ = docs
            cls.calls += 1
            raise ImportError("missing rank_bm25")

    monkeypatch.setattr("taxi_agent.retrieval.BM25Retriever", MissingBM25)

    retriever = SchemaRetriever(
        embedding_model=None,
        config=SchemaRetrieverConfig(top_k_tables=1, search_type="mmr", fetch_k=20),
    )
    tables = _sample_tables()

    retriever.refresh(tables)
    retriever.refresh(tables)

    assert MissingBM25.calls == 1
    assert retriever.keyword_retriever is None


def test_vector_retried_when_previous_build_failed() -> None:
    retriever = SchemaRetriever(
        embedding_model=FlakyEmbeddings(),
        config=SchemaRetrieverConfig(top_k_tables=1, search_type="mmr", fetch_k=20),
    )
    tables = _sample_tables()

    retriever.refresh(tables)
    assert retriever.retriever is None

    # Same schema fingerprint; refresh should retry vector initialization.
    retriever.refresh(tables)
    assert retriever.retriever is not None


def test_ensemble_retriever_preferred_when_available(monkeypatch) -> None:
    class FakeBM25Retriever:
        k = 0

        @classmethod
        def from_documents(cls, docs):
            _ = docs
            return cls()

        def invoke(self, question: str):
            _ = question
            return [
                Document(
                    page_content="Table: public.zones",
                    metadata={"full_name": "public.zones"},
                )
            ]

    class FakeEnsembleRetriever:
        instances = 0

        def __init__(self, retrievers, weights):
            _ = (retrievers, weights)
            FakeEnsembleRetriever.instances += 1

        def invoke(self, question: str):
            _ = question
            return [
                Document(
                    page_content="Table: public.taxi_trip_data",
                    metadata={"full_name": "public.taxi_trip_data"},
                )
            ]

    monkeypatch.setattr("taxi_agent.retrieval.BM25Retriever", FakeBM25Retriever)
    monkeypatch.setattr("taxi_agent.retrieval.EnsembleRetriever", FakeEnsembleRetriever)

    retriever = SchemaRetriever(
        embedding_model=StableEmbeddings(),
        config=SchemaRetrieverConfig(top_k_tables=1, search_type="mmr", fetch_k=20),
    )
    retriever.refresh(_sample_tables())

    selected = retriever.retrieve_tables("any question")
    assert selected[0].table_name == "taxi_trip_data"
    assert FakeEnsembleRetriever.instances == 1
