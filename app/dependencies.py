from functools import lru_cache
from typing import AsyncIterator

from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.config import Settings, get_settings
from app.infrastructure.azure_openai.embedding_client import AzureEmbeddingClient
from app.infrastructure.azure_openai.llm_client import AzureLlmClient
from app.infrastructure.parsers.markdown_parser import MarkdownParser
from app.infrastructure.parsers.pdf_parser import PdfParser
from app.infrastructure.postgres.bm25_repository import Bm25Repository
from app.infrastructure.postgres.document_registry import PostgresDocumentRegistry
from app.infrastructure.postgres.pgvector_repository import PgVectorRepository
from app.infrastructure.reranker.cross_encoder_reranker import CrossEncoderReranker


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(get_settings().database_url)


async def get_db() -> AsyncIterator[AsyncSession]:
    session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    async with session_factory() as session:
        yield session


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def get_answer_query_use_case(
    db: AsyncSession = Depends(get_db),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    settings: Settings = Depends(get_settings),
) -> AnswerQueryUseCase:
    return AnswerQueryUseCase(
        embedding_port=AzureEmbeddingClient(settings),
        vector_store=PgVectorRepository(db),
        lexical_search=Bm25Repository(db),
        reranker=reranker,
        llm=AzureLlmClient(settings),
    )


def get_ingest_document_use_case(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> IngestDocumentUseCase:
    return IngestDocumentUseCase(
        parsers={"md": MarkdownParser(), "pdf": PdfParser()},
        registry=PostgresDocumentRegistry(db),
        embedding_port=AzureEmbeddingClient(settings),
        vector_store=PgVectorRepository(db),
        lexical_search=Bm25Repository(db),
    )
