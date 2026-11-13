"""Store and search vectors using PostgreSQL + pgvector."""
import json
import math
from typing import Any
import psycopg
from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """Retain connection settings; validate model dimensions."""
        self.database_url = database_url
        self.dimensions = dimensions
        if dimensions <= 0:
            raise ValueError("Dimensions must be greater than 0")

    def _get_conn(self):
        """Helper method to create a database connection."""
        return psycopg.connect(self.database_url)

    def _vector_text(self, vector: list[float]) -> str:
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector must have {self.dimensions} dimensions")
        values = [float(value) for value in vector]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Vector values must all be finite numbers")
        return "[" + ",".join(repr(value) for value in values) + "]"

    def create_schema(self) -> None:
        """Create table with vector column and source/page/filter metadata."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id VARCHAR(255) PRIMARY KEY,
                        document_id VARCHAR(255),
                        source VARCHAR(255),
                        page INT,
                        strategy VARCHAR(50),
                        text TEXT,
                        metadata JSONB,
                        embedding vector({self.dimensions}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace one strategy's index atomically, making reruns idempotent."""
        if len(chunks) != len(vectors):
            raise ValueError("Số lượng chunks và vectors không khớp nhau.")
        vector_texts = [self._vector_text(vector) for vector in vectors]

        self.create_schema()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE strategy = %s;", (strategy,))
                for chunk, vector_text in zip(chunks, vector_texts):
                    cur.execute("""
                        INSERT INTO document_chunks (chunk_id, document_id, source, page, strategy, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding;
                    """, (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.source,
                        chunk.page,
                        strategy,
                        chunk.text,
                        json.dumps(chunk.metadata),
                        vector_text
                    ))

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL, rank by cosine distance, return top-k."""
        query_vector_text = self._vector_text(query_vector)
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self.create_schema()

        query = """
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            WHERE strategy = %s
        """
        params: list[Any] = [query_vector_text, strategy]

        if filters:
            for key, val in filters.items():
                query += f" AND metadata ->> %s = %s"
                params.extend([key, str(val)])

        query += " ORDER BY embedding <=> %s::vector LIMIT %s;"
        params.extend([query_vector_text, top_k])

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                results = []
                for row in cur.fetchall():
                    results.append(dict(zip(columns, row)))
                return results

    def count(self, strategy: str) -> int:
        """Return the number of indexed chunks for a strategy."""
        self.create_schema()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM document_chunks WHERE strategy = %s;",
                    (strategy,),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(f"Unable to count stored chunks for strategy {strategy}")
                return row[0]

    def backend_info(self) -> dict[str, str]:
        """Return the PostgreSQL and pgvector versions used by this store."""
        self.create_schema()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT current_setting('server_version'),
                           (SELECT extversion FROM pg_extension WHERE extname = 'vector')
                """)
                row = cur.fetchone()
                if row is None or row[1] is None:
                    raise RuntimeError("Unable to read PostgreSQL and pgvector versions")
                return {"postgresql_version": row[0], "pgvector_version": row[1]}