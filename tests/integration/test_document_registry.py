from datetime import date

import pytest

from app.domain.models import Document
from app.infrastructure.postgres.document_registry import PostgresDocumentRegistry


def make_document(
    product_ref: str, content_hash: str, document_type: str = "datasheet"
) -> Document:
    return Document(
        id=f"{product_ref}::{document_type}",
        title=f"Fiche {product_ref}",
        product_ref=product_ref,
        version="1",
        status="active",
        document_type=document_type,
        published_date=date(2026, 1, 1),
        source_path=f"{product_ref}.md",
        content_hash=content_hash,
    )


@pytest.mark.asyncio
async def test_get_active_hash_returns_none_when_no_document_registered(db_session):
    # Arrange
    registry = PostgresDocumentRegistry(db_session)

    # Act
    result = await registry.get_active_hash("REF-UNKNOWN", "datasheet")

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_register_then_get_active_hash_returns_the_registered_hash(db_session):
    # Arrange
    registry = PostgresDocumentRegistry(db_session)
    document = make_document("REF-1", "hash-1")

    # Act
    await registry.register(document)
    result = await registry.get_active_hash("REF-1", "datasheet")

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
        document_id="REF-2::datasheet",
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
    await registry.deprecate("REF-2", "datasheet")
    active_hash = await registry.get_active_hash("REF-2", "datasheet")
    refreshed_chunk = await db_session.get(ChunkRow, "chunk-1")

    # Assert
    assert active_hash is None
    assert refreshed_chunk.status == "deprecated"


@pytest.mark.asyncio
async def test_two_document_types_with_same_product_ref_do_not_collide(db_session):
    # Arrange
    registry = PostgresDocumentRegistry(db_session)
    datasheet = make_document("REF-9", "hash-datasheet", document_type="datasheet")
    manuel = make_document("REF-9", "hash-manuel", document_type="manuel")

    # Act
    await registry.register(datasheet)
    await registry.register(manuel)
    datasheet_hash = await registry.get_active_hash("REF-9", "datasheet")
    manuel_hash = await registry.get_active_hash("REF-9", "manuel")

    # Assert — both created independently with distinct hashes
    assert datasheet_hash == "hash-datasheet"
    assert manuel_hash == "hash-manuel"

    # Act — deprecating one must not affect the other
    await registry.deprecate("REF-9", "datasheet")
    datasheet_hash_after = await registry.get_active_hash("REF-9", "datasheet")
    manuel_hash_after = await registry.get_active_hash("REF-9", "manuel")

    # Assert
    assert datasheet_hash_after is None
    assert manuel_hash_after == "hash-manuel"
