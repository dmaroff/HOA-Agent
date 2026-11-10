from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class Page:
    page_no: int
    text: str



@dataclass(frozen=True, kw_only=True)
class Chunk:
    doc_id: str
    chunk_index: int
    page_no: int  # page the chunk starts on (use for citations)
    end_page_no: int
    heading: str
    text: str