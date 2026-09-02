from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Document
from app.infrastructure.postgres.models import ChunkRow, DocumentRow


class PostgresDocumentRegistry:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_active_hash(self, product_ref: str) -> str | None:
        row = await self._session.get(DocumentRow, product_ref)
        return row.content_hash if row is not None and row.status == "active" else None

    async def register(self, document: Document) -> None:
        row = await self._session.get(DocumentRow, document.product_ref)
        if row is None:
            row = DocumentRow(product_ref=document.product_ref)
            self._session.add(row)
        row.title = document.title
        row.version = document.version
        row.status = "active"
        row.document_type = document.document_type
        row.published_date = document.published_date
        row.source_path = document.source_path
        row.content_hash = document.content_hash
        await self._session.commit()

    async def deprecate(self, product_ref: str) -> None:
        await self._session.execute(
            update(DocumentRow)
            .where(DocumentRow.product_ref == product_ref)
            .values(status="deprecated")
        )
        await self._session.execute(
            update(ChunkRow).where(ChunkRow.product_ref == product_ref).values(status="deprecated")
        )
        await self._session.commit()
