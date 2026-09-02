import hashlib
import uuid
from dataclasses import dataclass, replace

from app.domain.chunking import chunk_sections
from app.domain.errors import UnsupportedFormatError
from app.domain.models import Chunk
from app.domain.ports import (
    DocumentParserPort,
    DocumentRegistryPort,
    EmbeddingPort,
    LexicalSearchPort,
    VectorStorePort,
)
from app.domain.versioning import resolve_ingest_action


@dataclass
class IngestResult:
    document_id: str
    chunk_count: int
    status: str  # "created" | "updated" | "unchanged"


@dataclass
class IngestDocumentUseCase:
    parsers: dict[str, DocumentParserPort]
    registry: DocumentRegistryPort
    embedding_port: EmbeddingPort
    vector_store: VectorStorePort
    lexical_search: LexicalSearchPort

    async def execute(self, raw_bytes: bytes, filename: str, document_type: str) -> IngestResult:
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        parser = self.parsers.get(extension)
        if parser is None:
            raise UnsupportedFormatError(f"unsupported file extension: '{extension}'")

        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        document, raw_sections = parser.parse(raw_bytes, document_type, filename)
        document = replace(document, content_hash=content_hash)

        existing_hash = await self.registry.get_active_hash(document.product_ref)
        action = resolve_ingest_action(existing_hash, content_hash)

        if action == "unchanged":
            return IngestResult(document_id=document.id, chunk_count=0, status="unchanged")

        if action == "updated":
            await self.vector_store.delete_by_document_id(document.id)
            await self.lexical_search.delete_by_document_id(document.id)
            await self.registry.deprecate(document.product_ref)

        chunk_candidates = chunk_sections(raw_sections)
        chunks = [
            Chunk(
                id=str(uuid.uuid4()),
                document_id=document.id,
                content=candidate.content,
                content_type=candidate.content_type,
                title=document.title,
                product_ref=document.product_ref,
                version=document.version,
                status="active",
                document_type=document.document_type,
                published_date=document.published_date,
                content_hash=document.content_hash,
                source_path=document.source_path,
            )
            for candidate in chunk_candidates
        ]

        embeddings = [await self.embedding_port.embed(chunk.content) for chunk in chunks]
        await self.vector_store.upsert(chunks, embeddings)
        await self.lexical_search.upsert(chunks)
        await self.registry.register(document)

        return IngestResult(document_id=document.id, chunk_count=len(chunks), status=action)
