import anthropic

from app.models import Chunk

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You answer homeowner questions about their HOA's architectural guidelines. "
    "Use only the document excerpts provided. "
    "Cite the page numbers you relied on, for example (page 9). "
    "If the excerpts do not contain enough information to answer the question, "
    "clearly say so instead of guessing. "
    "Do not assume something is permitted simply because it is not prohibited "
    "in the provided excerpts. "
    "Distinguish between HOA requirements and recommendations. "
    "Keep answers clear, concise, and easy for homeowners to understand."
)


def build_context(chunks: list[Chunk]) -> str:
    """Join chunks into one block, each labeled with its source pages."""
    sections = []
    for chunk in chunks:
        if chunk.page_no == chunk.end_page_no:
            label = f"Page {chunk.page_no}"
        else:
            label = f"Pages {chunk.page_no}-{chunk.end_page_no}"
        sections.append(f"[{label}]\n{chunk.text}")
    return "\n\n".join(sections)


def answer_question(chunks: list[Chunk], question: str) -> str:
    # Reads the API key from the ANTHROPIC_API_KEY environment variable.
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Document excerpts:\n\n{build_context(chunks)}\n\nQuestion: {question}",
        }],
    )
    return "".join(block.text for block in response.content if block.type == "text")
