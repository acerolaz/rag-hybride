from datetime import date

import pytest

from app.domain.models import Document
from app.infrastructure.postgres.document_registry import PostgresDocumentRegistry


def make_document(product_ref: str, content_hash: str) -> Document:
    return Document(
        id=product_ref,
        title=f"Fiche {product_ref}",
        product_ref=product_ref,
        version="1",
        status="active",
        document_type="datasheet",
        published_date=date(2026, 1, 1),
        source_path=f"{product_ref}.md",
        content_hash=content_hash,
    )


@pytest.mark.asyncio
async def test_get_active_hash_returns_none_when_no_document_registered(db_session):
    # Arrange
    registry = PostgresDocumentRegistry(db_session)

    # Act
    result = await registry.get_active_hash("REF-UNKNOWN")

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_register_then_get_active_hash_returns_the_registered_hash(db_session):
    # Arrange
    registry = PostgresDocumentRegistry(db_session)
    document = make_document("REF-1", "hash-1")

    # Act
    await registry.register(document)
    result = await registry.get_active_hash("REF-1")

    # Assert
    assert result == "hash-1"


@pytest.mark.asyncio
async def test_deprecate_marks_document_and_its_chunks_as_deprecated(db_session):
    # Arrange
    from app.infrastructure.postgres.models import ChunkRow

    registry = PostgresDocumentRegistry(db_session)
    document = make_document("REF-2", "hash-2")
    await registry.register(document)
    chunk_row = ChunkRow(
        id="chunk-1",
        document_id="REF-2",
        content="x",
        content_type="text",
        title="t",
        product_ref="REF-2",
        version="1",
        status="active",
        document_type="datasheet",
        published_date=date(2026, 1, 1),
        content_hash="hash-2",
        source_path="p",
    )
    db_session.add(chunk_row)
    await db_session.commit()

    # Act
    await registry.deprecate("REF-2")
    active_hash = await registry.get_active_hash("REF-2")

    # Assert
    assert active_hash is None
