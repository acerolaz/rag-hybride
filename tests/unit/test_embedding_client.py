from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import Settings
from app.infrastructure.azure_openai.embedding_client import AzureEmbeddingClient


def make_settings() -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://u:p@localhost/db",
        azure_openai_api_key="key",
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_embedding_deployment="text-embedding-3-large",
        azure_openai_chat_deployment="gpt-4o",
    )


@pytest.mark.asyncio
async def test_embed_returns_the_vector_from_the_first_response_item():
    # Arrange
    with patch(
        "app.infrastructure.azure_openai.embedding_client.AsyncAzureOpenAI"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        client = AzureEmbeddingClient(make_settings())

        # Act
        vector = await client.embed("tension nominale")

        # Assert
        assert vector == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-large", input="tension nominale"
        )
