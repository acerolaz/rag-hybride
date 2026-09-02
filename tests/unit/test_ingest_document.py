from datetime import date

import pytest

from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.domain.chunking import RawSection
from app.domain.errors import UnsupportedFormatError
from app.domain.models import Document


def make_document(product_ref: str) -> Document:
    return Document(
        id=product_ref, title=f"Fiche {product_ref}", product_ref=product_ref, version="1",
        status="active", document_type="datasheet", published_date=date(2026, 1, 1),
        source_path=f"{product_ref}.md", content_hash="",
    )


class FakeParser:
    def __init__(self, document: Document, sections: list[RawSection]):
        self._document = document
        self._sections = sections

    def parse(self, raw_bytes, document_type, source_path):
        return self._document, self._sections


class FakeRegistry:
    def __init__(self, active_hash: str | None = None):
        self.active_hash = active_hash
        self.registered: list[Document] = []
        self.deprecated: list[str] = []

    async def get_active_hash(self, product_ref: str) -> str | None:
        return self.active_hash

    async def register(self, document: Document) -> None:
        self.registered.append(document)

    async def deprecate(self, product_ref: str) -> None:
        self.deprecated.append(product_ref)


class FakeEmbeddingPort:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self):
        self.upserted: list = []
        self.deleted: list[str] = []

    async def upsert(self, chunks, embeddings) -> None:
        self.upserted.append(chunks)

    async def search(self, embedding, top_k, product_ref=None):
        raise NotImplementedError

    async def delete_by_document_id(self, document_id: str) -> None:
        self.deleted.append(document_id)


class FakeLexicalSearch:
    def __init__(self):
        self.upserted: list = []
        self.deleted: list[str] = []

    async def upsert(self, chunks) -> None:
        self.upserted.append(chunks)

    async def search(self, query, top_k, product_ref=None):
        raise NotImplementedError

    async def delete_by_document_id(self, document_id: str) -> None:
        self.deleted.append(document_id)


@pytest.mark.asyncio
async def test_new_document_is_created_and_indexed():
    # Arrange
    document = make_document("REF-1")
    sections = [RawSection(content="Contenu de la fiche produit REF-1", content_type="text")]
    registry = FakeRegistry(active_hash=None)
    vector_store = FakeVectorStore()
    lexical_search = FakeLexicalSearch()
    use_case = IngestDocumentUseCase(
        parsers={"md": FakeParser(document, sections)},
        registry=registry,
        embedding_port=FakeEmbeddingPort(),
        vector_store=vector_store,
        lexical_search=lexical_search,
    )

    # Act
    result = await use_case.execute(b"raw content", "REF-1.md", "datasheet")

    # Assert
    assert result.status == "created"
    assert result.chunk_count == 1
    assert len(registry.registered) == 1
    assert len(vector_store.upserted) == 1


@pytest.mark.asyncio
async def test_unchanged_hash_is_a_noop():
    # Arrange
    document = make_document("REF-2")
    sections = [RawSection(content="Contenu identique", content_type="text")]
    import hashlib

    raw_bytes = b"same content"
    same_hash = hashlib.sha256(raw_bytes).hexdigest()
    registry = FakeRegistry(active_hash=same_hash)
    vector_store = FakeVectorStore()
    use_case = IngestDocumentUseCase(
        parsers={"md": FakeParser(document, sections)},
        registry=registry,
        embedding_port=FakeEmbeddingPort(),
        vector_store=vector_store,
        lexical_search=FakeLexicalSearch(),
    )

    # Act
    result = await use_case.execute(raw_bytes, "REF-2.md", "datasheet")

    # Assert
    assert result.status == "unchanged"
    assert vector_store.upserted == []


@pytest.mark.asyncio
async def test_changed_hash_deprecates_old_version_before_reindexing():
    # Arrange
    document = make_document("REF-3")
    sections = [RawSection(content="Nova versão do conteúdo", content_type="text")]
    registry = FakeRegistry(active_hash="old-hash")
    vector_store = FakeVectorStore()
    use_case = IngestDocumentUseCase(
        parsers={"md": FakeParser(document, sections)},
        registry=registry,
        embedding_port=FakeEmbeddingPort(),
        vector_store=vector_store,
        lexical_search=FakeLexicalSearch(),
    )

    # Act
    result = await use_case.execute(b"new content", "REF-3.md", "datasheet")

    # Assert
    assert result.status == "updated"
    assert registry.deprecated == ["REF-3"]
    assert vector_store.deleted == ["REF-3"]


@pytest.mark.asyncio
async def test_unsupported_extension_raises():
    # Arrange
    use_case = IngestDocumentUseCase(
        parsers={"md": FakeParser(make_document("REF-4"), [])},
        registry=FakeRegistry(),
        embedding_port=FakeEmbeddingPort(),
        vector_store=FakeVectorStore(),
        lexical_search=FakeLexicalSearch(),
    )

    # Act / Assert
    with pytest.raises(UnsupportedFormatError):
        await use_case.execute(b"data", "notes.docx", "datasheet")
