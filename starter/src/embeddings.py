"""Choose a real embedding provider (API or local) and implement this contract."""
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class MyEmbedder:
    """Real provider adapter using google-genai with gemini-embedding-001."""

    def __init__(self, model_name: str = "gemini-embedding-001") -> None:
        self.model_name = model_name
        self.dimensions = 768
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini embeddings")
        self.client = genai.Client(
            api_key=api_key
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=self.dimensions)
            )
            if not response.embeddings or len(response.embeddings) != len(batch):
                raise RuntimeError("Gemini returned an unexpected number of document embeddings")
            embeddings.extend(embedding.values for embedding in response.embeddings)
        return embeddings

    def embed_query(self, question: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=question,
            config=types.EmbedContentConfig(output_dimensionality=self.dimensions)
        )
        if not response.embeddings:
            raise RuntimeError("Gemini returned no query embedding")
        return response.embeddings[0].values