from datetime import date
from typing import Annotated

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 1536
PrimaryKeyStr = Annotated[str, mapped_column(String, primary_key=True)]


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    product_ref: Mapped[PrimaryKeyStr]
    title: Mapped[str]
    version: Mapped[str]
    status: Mapped[str] = mapped_column(String, index=True)
    document_type: Mapped[str]
    published_date: Mapped[date]
    source_path: Mapped[str]
    content_hash: Mapped[str]


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[PrimaryKeyStr]
    document_id: Mapped[str] = mapped_column(String, index=True)
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
    search_vector: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)
