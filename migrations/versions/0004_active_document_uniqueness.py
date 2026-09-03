"""Enforce at most one active document per product_ref and document_type

Ensures only one document can have status='active' for a given (product_ref, document_type)
combination, preventing concurrent ingests from leaving multiple active versions.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-03
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "uq_documents_active_ref_type",
        "documents",
        ["product_ref", "document_type"],
        unique=True,
        postgresql_where="status = 'active'",
    )


def downgrade() -> None:
    op.drop_index("uq_documents_active_ref_type", table_name="documents")
