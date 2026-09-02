from openai import AsyncAzureOpenAI

from app.config import Settings


class AzureEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        self._deployment = settings.azure_openai_embedding_deployment

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=self._deployment, input=text)
        return list(response.data[0].embedding)
