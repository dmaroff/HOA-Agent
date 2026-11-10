from dataclasses import asdict, dataclass
from typing import cast
import numpy as np
from sentence_transformers import SentenceTransformer
from app.models import Chunk


model = SentenceTransformer("BAAI/bge-base-en-v1.5")

print(f"Model max sequence length: {model.max_seq_length}")

@dataclass(frozen=True, kw_only=True)
class EmbeddedChunk(Chunk):
    embedding: list[float]


def create_embeddings(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    embeddings = cast(
        np.ndarray,
        model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            convert_to_tensor=False,
            output_value="sentence_embedding"
        )
    )

    embedded_chunks: list[EmbeddedChunk] = []

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        embedded_chunks.append(
            EmbeddedChunk(
                **asdict(chunk),
                embedding=embedding.tolist()
            )
        )

    for chunk in chunks:
        token_count = len(
            model.tokenizer.encode(chunk.text)
        )
        print(
            f"Chunk {chunk.chunk_index}: "
            f"{token_count} tokens"
        )

    return embedded_chunks

def search(question: str, embedded_chunks: list[EmbeddedChunk]) -> list[EmbeddedChunk]:

    """Find the chunk most similar to the question; return its score and that chunk
    together with the chunk before and after it (same document), in reading order."""
    question_embedding = model.encode(question, normalize_embeddings=True, convert_to_numpy=True)
    chunk_matrix = np.array([chunk.embedding for chunk in embedded_chunks])

    # Embeddings are normalized, so the dot product is the cosine similarity.
    scores = chunk_matrix @ question_embedding
    best = int(np.argmax(scores))

    doc_id = embedded_chunks[best].doc_id
    neighbors = embedded_chunks[max(best - 1, 0):best + 2]
    return [chunk for chunk in neighbors if chunk.doc_id == doc_id]
