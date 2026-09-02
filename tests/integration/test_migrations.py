"""Migrations are exercised as migrations here.

The shared `db_session` fixture builds the schema with
`Base.metadata.create_all`, which never runs Alembic — so nothing else in the
suite would notice a broken revision. These tests drive `alembic upgrade head`
against a real pgvector container instead.
"""

import asyncio
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INSERT_CHUNK = sa.text(
    """
    INSERT INTO chunks (
        id, document_id, content, content_type, title, product_ref,
        version, status, document_type, published_date, content_hash, source_path
    ) VALUES (
        'chunk-1', 'doc-1', 'La pompe REF-8842 nécessite un joint torique.',
        'text', 'Fiche REF-8842', 'REF-8842', '1', 'active', 'datasheet',
        DATE '2026-01-01', 'hash-1', '/corpus/ref-8842.md'
    )
    """
)


def _alembic_config(url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    return config


async def _fetch_all(url: str, statements: list[sa.TextClause]) -> list[list[Any]]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return [list(await connection.execute(stmt)) for stmt in statements]
    finally:
        await engine.dispose()


async def _execute(url: str, statement: sa.TextClause) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(statement)
    finally:
        await engine.dispose()


@pytest.fixture
def migration_db_url(postgres_container: Any) -> Iterator[str]:
    """A dedicated, empty database per test, on the shared container.

    Migrations must not run against the database the ORM-based fixtures use,
    or the two schema-creation paths would fight over the same tables.
    """
    admin_url = postgres_container.get_connection_url()
    base, _, _ = admin_url.rpartition("/")
    name = f"migration_check_{uuid.uuid4().hex[:8]}"

    async def create() -> None:
        engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as connection:
                await connection.execute(sa.text(f'CREATE DATABASE "{name}"'))
        finally:
            await engine.dispose()

    async def drop() -> None:
        engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as connection:
                await connection.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}"'))
        finally:
            await engine.dispose()

    asyncio.run(create())
    try:
        yield f"{base}/{name}"
    finally:
        asyncio.run(drop())


def test_upgrade_head_installs_extension_and_both_tables(migration_db_url: str) -> None:
    # Arrange
    config = _alembic_config(migration_db_url)

    # Act
    command.upgrade(config, "head")

    # Assert
    extensions, tables = asyncio.run(
        _fetch_all(
            migration_db_url,
            [
                sa.text("SELECT extname FROM pg_extension WHERE extname = 'vector'"),
                sa.text(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
                ),
            ],
        )
    )
    assert [row[0] for row in extensions] == ["vector"]
    assert {row[0] for row in tables} >= {"chunks", "documents"}


def test_upgrade_head_creates_gin_and_hnsw_retrieval_indexes(migration_db_url: str) -> None:
    # Arrange
    config = _alembic_config(migration_db_url)

    # Act
    command.upgrade(config, "head")

    # Assert — both hybrid retrieval paths must be index-backed, and by the
    # right access method: a btree on either column would be useless.
    (index_rows,) = asyncio.run(
        _fetch_all(
            migration_db_url,
            [
                sa.text(
                    """
                    SELECT idx_class.relname, method.amname
                    FROM pg_index AS idx
                    JOIN pg_class AS idx_class ON idx_class.oid = idx.indexrelid
                    JOIN pg_class AS tbl_class ON tbl_class.oid = idx.indrelid
                    JOIN pg_am AS method ON method.oid = idx_class.relam
                    WHERE tbl_class.relname = 'chunks'
                    """
                )
            ],
        )
    )
    methods_by_index = dict(index_rows)
    assert methods_by_index.get("ix_chunks_search_vector_gin") == "gin"
    assert methods_by_index.get("ix_chunks_embedding_hnsw") == "hnsw"


def test_search_vector_is_generated_by_postgres_on_insert(migration_db_url: str) -> None:
    # Arrange
    command.upgrade(_alembic_config(migration_db_url), "head")

    # Act — the insert never mentions search_vector.
    asyncio.run(_execute(migration_db_url, INSERT_CHUNK))

    # Assert
    (generated_row,), (search_vector_row,) = asyncio.run(
        _fetch_all(
            migration_db_url,
            [
                sa.text(
                    "SELECT is_generated FROM information_schema.columns "
                    "WHERE table_name = 'chunks' AND column_name = 'search_vector'"
                ),
                sa.text("SELECT search_vector FROM chunks WHERE id = 'chunk-1'"),
            ],
        )
    )
    assert generated_row[0] == "ALWAYS"
    # French stemming reduces "pompe" to "pomp", proving the configured
    # dictionary — not the default one — produced the vector.
    assert "pomp" in search_vector_row[0]


def test_downgrade_to_base_drops_tables_but_keeps_the_extension(migration_db_url: str) -> None:
    # Arrange
    config = _alembic_config(migration_db_url)
    command.upgrade(config, "head")

    # Act
    command.downgrade(config, "base")

    # Assert — dropping a database-wide extension is not this schema's call.
    tables, extensions = asyncio.run(
        _fetch_all(
            migration_db_url,
            [
                sa.text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"),
                sa.text("SELECT extname FROM pg_extension WHERE extname = 'vector'"),
            ],
        )
    )
    remaining = {row[0] for row in tables}
    assert "chunks" not in remaining
    assert "documents" not in remaining
    assert [row[0] for row in extensions] == ["vector"]
