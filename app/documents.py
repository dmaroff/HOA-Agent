import re
from dataclasses import dataclass
from typing import Final

import pymupdf

from app.models import Chunk, Page

CHUNK_SIZE: Final[int] = 2000
OVERLAP: Final[int] = 300

# "STANDARD NUMBER 7" or a roman-numeral section such as "III." (upper case only,
# so the mixed-case table of contents is not mistaken for headings).
HEADING_RE: Final = re.compile(r"^(STANDARD NUMBER \d+|[IVX]+\.)\s*$")
# A numbered or lettered list marker (1. / 12. / A.) starts a new paragraph instead of
# continuing the previous line.
LIST_ITEM_RE: Final = re.compile(r"^(\d+|[A-Za-z])\.\s")
PAGE_NUMBER_RE: Final = re.compile(r"^\d{1,3}$")
# Split after ". ! ?" but not after list markers ("3.") or abbreviations ("e.g.").
SENTENCE_END_RE: Final = re.compile(r"(?:(?<=[a-z]{2}[.!?])|(?<=\)[.!?]))\s+(?=[A-Z0-9(])")


@dataclass(frozen=True, kw_only=True)
class _Segment:
    """A sentence (or a whole heading) that is never split further."""
    page_no: int
    end_page_no: int
    heading: str
    text: str
    is_heading: bool = False
    starts_paragraph: bool = False
    starts_list_item: bool = False


def extract_pages(pdf_path: str) -> list[Page]:
    pages: list[Page] = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            pages.append(
                Page(page_no=page_number, text=page.get_text("text"))
            )
    return pages


def _page_paragraphs(text: str) -> list[str]:
    """Rebuild paragraphs from PDF line breaks, dropping bare page numbers."""
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if PAGE_NUMBER_RE.match(line):
            continue
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if current and (LIST_ITEM_RE.match(line) or HEADING_RE.match(line)):
            paragraphs.append(" ".join(current))
            current = []
        current.append(re.sub(r"\s+", " ", line))
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _build_segments(pages: list[Page]) -> list[_Segment]:
    """Turn pages into sentence-level segments tagged with page and section heading."""
    # Paragraphs as [page_no, end_page_no, text]; a paragraph that runs over a page
    # break is merged so its sentences stay together.
    paragraphs: list[list] = []
    for page in pages:
        for index, paragraph in enumerate(_page_paragraphs(page.text)):
            previous = paragraphs[-1] if paragraphs else None
            if (
                previous is not None
                and index == 0
                and paragraph[0].islower()
                and not previous[2].endswith((".", "!", "?", ":", ";"))
                and not HEADING_RE.match(previous[2])
            ):
                previous[1] = page.page_no
                previous[2] += " " + paragraph
            else:
                paragraphs.append([page.page_no, page.page_no, paragraph])

    segments: list[_Segment] = []
    heading = ""
    expect_title = False
    for page_no, end_page_no, text in paragraphs:
        if HEADING_RE.match(text):
            heading = text
            expect_title = True
            segments.append(_Segment(
                page_no=page_no,
                end_page_no=page_no,
                heading=heading,
                text=text,
                is_heading=True,
                starts_paragraph=True)
            )
            continue
        if expect_title and not LIST_ITEM_RE.match(text) and len(text) < 120:
            # Title line that follows "STANDARD NUMBER n" (maybe on the next page).
            heading = f"{heading} - {text}"
            segments.append(_Segment(page_no=page_no, end_page_no=page_no, heading=heading,
                                     text=text, is_heading=True))
            expect_title = False
            continue
        expect_title = False
        sentences = SENTENCE_END_RE.split(text)
        for i, sentence in enumerate(sentences):
            segments.append(_Segment(
                page_no=page_no,
                end_page_no=end_page_no if i == len(sentences) - 1 else page_no,
                heading=heading,
                text=sentence,
                starts_paragraph=i == 0,
                starts_list_item=i == 0 and bool(LIST_ITEM_RE.match(sentence)),
            ))
    return _split_oversized(segments)

def _split_oversized(segments: list[_Segment]) -> list[_Segment]:
    """Break any sentence longer than a chunk at word boundaries (last resort)."""
    result: list[_Segment] = []
    for segment in segments:
        if len(segment.text) <= CHUNK_SIZE:
            result.append(segment)
            continue
        words = segment.text.split(" ")
        piece = ""
        first = True
        for word in words:
            if piece and len(piece) + 1 + len(word) > CHUNK_SIZE:
                result.append(_replace_text(segment, piece, starts_paragraph=first and segment.starts_paragraph))
                piece, first = word, False
            else:
                piece = f"{piece} {word}" if piece else word
        result.append(_replace_text(segment, piece, starts_paragraph=False))
    return result

def _replace_text(segment: _Segment, text: str, *, starts_paragraph: bool) -> _Segment:
    return _Segment(page_no=segment.page_no, end_page_no=segment.end_page_no,
                    heading=segment.heading, text=text, is_heading=segment.is_heading,
                    starts_paragraph=starts_paragraph)


def _join(segments: list[_Segment]) -> str:
    text = ""
    for segment in segments:
        if not text:
            text = segment.text
        elif segment.starts_list_item or segment.is_heading:
            text += "\n" + segment.text
        elif segment.starts_paragraph:
            text += "\n" + segment.text
        else:
            text += " " + segment.text
    return text

def _heading_prefix(segments: list[_Segment]) -> str:
    """Heading to repeat on a chunk that doesn't already begin with its section heading."""
    first = segments[0]
    if not first.heading or first.is_heading:
        return ""
    return first.heading + "\n"

def _overlap_tail(segments: list[_Segment]) -> list[_Segment]:
    """Trailing whole sentences totaling at most OVERLAP characters."""
    tail: list[_Segment] = []
    size = 0
    for segment in reversed(segments):
        if segment.is_heading:
            break
        if size + len(segment.text) + 1 > OVERLAP:
            break
        tail.insert(0, segment)
        size += len(segment.text) + 1
    return tail

def chunk_text(pages: list[Page], doc_id: str = "") -> list[Chunk]:
    segments = _build_segments(pages)
    chunks: list[Chunk] = []

    def emit(members: list[_Segment]) -> None:
        text = _heading_prefix(members) + _join(members)
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_index=len(chunks),
            page_no=members[0].page_no,
            end_page_no=max(s.end_page_no for s in members),
            heading=members[-1].heading if members[0].is_heading else members[0].heading,
            text=text,
        ))

    def size_of(members: list[_Segment]) -> int:
        return len(_heading_prefix(members)) + len(_join(members))

    group: list[_Segment] = []
    fresh = 0  # segments at the end of `group` that are not overlap from the previous chunk
    for segment in segments:
        if group and size_of(group + [segment]) > CHUNK_SIZE:
            # Never leave a heading stranded at the end of a chunk.
            carry: list[_Segment] = []
            while len(group) > 1 and group[-1].is_heading:
                carry.insert(0, group.pop())
            if fresh > 0 or not chunks:
                emit(group)
            overlap = _overlap_tail(group)
            group = overlap + carry
            if size_of(group + [segment]) > CHUNK_SIZE:
                group = carry  # overlap doesn't fit alongside this sentence; drop it
            fresh = len(carry)
        group.append(segment)
        fresh += 1
    if group and fresh > 0:
        emit(group)
    return chunks