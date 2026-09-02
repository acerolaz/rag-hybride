import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.schemas.ingest import IngestResponse
from app.application.use_cases.ingest_document import IngestDocumentUseCase
from app.dependencies import get_ingest_document_use_case
from app.domain.errors import UnparsableDocumentError, UnsupportedFormatError

router = APIRouter(prefix="/api/v1", tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    document_type: str = Form(...),
    file: UploadFile = File(...),
    use_case: IngestDocumentUseCase = Depends(get_ingest_document_use_case),
) -> IngestResponse:
    """Ingest a source document (Markdown or PDF) into the dual dense/lexical index."""
    raw_bytes = await file.read()
    try:
        result = await use_case.execute(raw_bytes, file.filename or "", document_type)
    except UnsupportedFormatError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "UNSUPPORTED_FORMAT",
                "message": str(exc),
                "correlation_id": str(uuid.uuid4()),
            },
        ) from exc
    except UnparsableDocumentError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "UNPARSABLE_DOCUMENT",
                "message": str(exc),
                "correlation_id": str(uuid.uuid4()),
            },
        ) from exc
    return IngestResponse(
        document_id=result.document_id, chunk_count=result.chunk_count, status=result.status
    )
