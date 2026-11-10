from pathlib import Path
from openai import OpenAI


import app.documents as docs
import app.embeddings as emb
import app.llm_service as llm


def generate_embedding(text: str) -> list[float]:
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def process_pdf():
    pdf_path = "/home/dan/Downloads/arc-guidelines.pdf"
    pages = docs.extract_pages(pdf_path)
    chunks = docs.chunk_text(pages, doc_id=Path(pdf_path).stem)
    embedded_chunks = emb.create_embeddings(chunks)
    for chunk in embedded_chunks:
        print(
            f"Chunk {chunk.chunk_index} | "
            f"Pages {chunk.page_no}-{chunk.end_page_no} | "
            f"Chars {len(chunk.text)} | "
            f"Embedding dimensions {len(chunk.embedding)}"
        )
    return embedded_chunks


def ask_questions(embedded_chunks):
    while True:
        question = input("\nQuestion (blank to quit): ").strip()
        if not question:
            break
        chunks = emb.search(question, embedded_chunks)
        answer = llm.answer_question(chunks, question)
        print(f"{answer}\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    embedded_chunks = process_pdf()
    ask_questions(embedded_chunks)
