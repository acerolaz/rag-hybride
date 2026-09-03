from datetime import date
from typing import Annotated

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import literal_column

EMBEDDING_DIM = 1536

# Text search configuration used for the lexical (BM25-style) index. Declared
# once here so the generated column, its GIN index and the query-time
# `plainto_tsquery` in Bm25Repository can never drift apart.
TSVECTOR_CONFIG = "french"

PrimaryKeyStr = Annotated[str, mapped_column(String, primary_key=True)]


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # One row per version of a document. A composite *primary* key of
        # (product_ref, document_type) could only ever hold the current
        # version, so a superseded one had nowhere to live.
        UniqueConstraint(
            "product_ref",
            "document_type",
            "version",
            name="uq_documents_product_ref_document_type_version",
        ),
        # Serves the registry's hot lookup: the active version of one
        # product's document of a given type.
        Index("ix_documents_ref_type_status", "product_ref", "document_type", "status"),
        # Enforce only one active version per (product_ref, document_type) to prevent
        # concurrent ingests from leaving multiple active documents.
        Index(
            "uq_documents_active_ref_type",
            "product_ref",
            "document_type",
            unique=True,
            postgresql_where=literal_column("status = 'active'"),
        ),
    )

    id: Mapped[PrimaryKeyStr]
    product_ref: Mapped[str]
    document_type: Mapped[str]
    title: Mapped[str]
    version: Mapped[str]
    status: Mapped[str] = mapped_column(String, index=True)
    published_date: Mapped[date]
    source_path: Mapped[str]
    content_hash: Mapped[str]


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        # Lexical retrieval: without a GIN index every `@@` match degrades to a
        # sequential scan over the whole corpus.
        Index("ix_chunks_search_vector_gin", "search_vector", postgresql_using="gin"),
        # Dense retrieval: HNSW with cosine ops, matching PgVectorRepository's
        # `embedding.cosine_distance(...)` ordering.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": "16", "ef_construction": "64"},
        ),
    )

    id: Mapped[PrimaryKeyStr]
    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
            name="fk_chunks_document_id_documents",
        ),
        index=True,
    )
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str]
    title: Mapped[str]
    product_ref: Mapped[str] = mapped_column(String, index=True)
    version: Mapped[str]
    status: Mapped[str] = mapped_column(String, index=True)
    document_type: Mapped[str]
    published_date: Mapped[date]
    content_hash: Mapped[str]
    source_path: Mapped[str]
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    # Maintained by Postgres, not by the repository: a STORED generated column
    # cannot drift from `content` the way a hand-written assignment can.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(f"to_tsvector('{TSVECTOR_CONFIG}'::regconfig, content)", persisted=True),
        nullable=True,
    )
