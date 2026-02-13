from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever

from .schema import TableSchema

try:
    from langchain_community.retrievers import BM25Retriever
except Exception:  # pragma: no cover
    BM25Retriever = None  # type: ignore

try:
    from langchain_classic.retrievers import EnsembleRetriever
except Exception:  # pragma: no cover
    EnsembleRetriever = None  # type: ignore


@dataclass
class SchemaRetrieverConfig:
    top_k_tables: int
    search_type: str
    fetch_k: int


class SchemaRetriever:
    def __init__(
        self,
        embedding_model: Optional[Embeddings],
        config: SchemaRetrieverConfig,
    ):
        self.logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model
        self.config = config
        self.table_map: Dict[str, TableSchema] = {}
        self.retriever: Optional[VectorStoreRetriever] = None
        self.keyword_retriever = None
        self.ensemble_retriever = None
        self.schema_fingerprint: str = ""
        self._bm25_dependency_missing = False

    def _make_schema_fingerprint(self, tables: Sequence[TableSchema]) -> str:
        parts = []
        for table in tables:
            cols = ",".join(f"{c.column_name}:{c.data_type}" for c in table.columns)
            parts.append(f"{table.full_name}|{cols}")
        return "||".join(parts)

    def _table_to_document(self, table: TableSchema) -> Document:
        column_lines = [f"- {c.column_name} ({c.data_type})" for c in table.columns]
        content = (
            f"Table: {table.full_name}\n"
            "Columns:\n"
            + "\n".join(column_lines)
        )
        return Document(
            page_content=content,
            metadata={
                "full_name": table.full_name.lower(),
                "table_name": table.table_name.lower(),
                "table_schema": table.table_schema.lower(),
            },
        )

    def refresh(self, tables: Sequence[TableSchema]) -> None:
        self.table_map = {t.full_name.lower(): t for t in tables}
        fingerprint = self._make_schema_fingerprint(tables)
        if fingerprint == self.schema_fingerprint:
            has_vector = self.retriever is not None
            has_keyword = self.keyword_retriever is not None
            vector_expected = self.embedding_model is not None
            keyword_expected = (
                BM25Retriever is not None and not self._bm25_dependency_missing
            )
            vector_ready = (not vector_expected) or has_vector
            keyword_ready = (not keyword_expected) or has_keyword
            if vector_ready and keyword_ready:
                return

        self.schema_fingerprint = fingerprint
        docs = [self._table_to_document(t) for t in tables]

        if BM25Retriever is not None and not self._bm25_dependency_missing:
            try:
                bm25 = BM25Retriever.from_documents(docs)
                bm25.k = self.config.top_k_tables
                self.keyword_retriever = bm25
            except Exception as exc:
                if isinstance(exc, ImportError):
                    self._bm25_dependency_missing = True
                self.logger.warning(
                    "BM25 retriever unavailable; fallback to vector-only retrieval: %s",
                    exc,
                )
                self.keyword_retriever = None
        else:
            self.keyword_retriever = None

        if not self.embedding_model:
            self.retriever = None
        else:
            try:
                vectorstore = InMemoryVectorStore(self.embedding_model)
                ids = [t.full_name.lower() for t in tables]
                vectorstore.add_documents(docs, ids=ids)

                search_kwargs = {"k": self.config.top_k_tables}
                if self.config.search_type == "mmr":
                    search_kwargs["fetch_k"] = max(
                        self.config.fetch_k,
                        self.config.top_k_tables * 4,
                    )

                self.retriever = vectorstore.as_retriever(
                    search_type=self.config.search_type,
                    search_kwargs=search_kwargs,
                )
            except Exception as exc:
                # If embeddings are unavailable at runtime, keep keyword retriever only.
                self.logger.warning(
                    "Vector retriever unavailable; fallback to BM25-only retrieval: %s",
                    exc,
                )
                self.retriever = None

        self.ensemble_retriever = None
        if EnsembleRetriever is not None:
            retrievers = []
            if self.retriever is not None:
                retrievers.append(self.retriever)
            if self.keyword_retriever is not None:
                retrievers.append(self.keyword_retriever)

            if len(retrievers) >= 2:
                try:
                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=retrievers,
                        weights=[1.0] * len(retrievers),
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Ensemble retriever unavailable; fallback to independent retrieval: %s",
                        exc,
                    )
                    self.ensemble_retriever = None

    def retrieve_tables(self, question: str) -> List[TableSchema]:
        if not self.table_map:
            return []

        docs: List[Document] = []
        if self.ensemble_retriever is not None:
            try:
                docs = list(self.ensemble_retriever.invoke(question))
            except Exception:
                docs = []
        else:
            if self.retriever is not None:
                try:
                    docs.extend(self.retriever.invoke(question))
                except Exception:
                    pass

            if self.keyword_retriever is not None:
                try:
                    docs.extend(self.keyword_retriever.invoke(question))
                except Exception:
                    pass

        if not docs:
            return list(self.table_map.values())[: self.config.top_k_tables]

        selected: List[TableSchema] = []
        seen = set()
        for doc in docs:
            key = str(doc.metadata.get("full_name", "")).lower()
            if key in seen:
                continue
            table = self.table_map.get(key)
            if table is None:
                continue
            seen.add(key)
            selected.append(table)
            if len(selected) >= self.config.top_k_tables:
                break

        if selected:
            return selected

        return list(self.table_map.values())[: self.config.top_k_tables]
