"""
Full-text document parser utility.

Spec: docs/07_EXTRACTION_STAGE.md §4

Parses XML (primary) or PDF (fallback via vision LLM) into a DocumentMap.
The DocumentMap schema is identical regardless of source format.
Downstream extraction nodes never know whether input was XML or PDF.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, TYPE_CHECKING

from lxml import etree

from src.schemas.extraction import (
    ArmInfo, DocumentMap, TextChunk, TimepointInfo,
)

if TYPE_CHECKING:
    from src.engine.agents import ExecutorAgent

logger = logging.getLogger("autosr.text_parser")

# ─────────────────────────────────────────────────────────────────────────────
# Section detection
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_KEYWORDS = {
    "abstract":    ["abstract"],
    "methods":     ["method", "patients", "participants", "study design",
                    "material", "setting", "procedure"],
    "results":     ["result", "finding", "outcome"],
    "discussion":  ["discussion", "conclusion", "interpretation", "implication"],
}


def _detect_section(title: str) -> str:
    """Map a section title string to a normalized section label."""
    t = title.lower().strip()
    for section, keywords in _SECTION_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return section.capitalize()
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
# XML parser (primary path — PubMed Central / NLM XML)
# ─────────────────────────────────────────────────────────────────────────────

def parse_xml(pmid: str, xml_path: str) -> DocumentMap:
    """
    Parse a PubMed Central XML file into a DocumentMap.

    Section extraction:
    - Abstract, Methods, Results, Discussion, Tables, Figure Captions
    Each <p> → one paragraph TextChunk.
    Each <table-wrap> → one table TextChunk.
    chunk_id format: "{Section}_{Type}_{index}"
    """
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
    except etree.XMLSyntaxError as exc:
        logger.error("[TextParser] XML parse error for %s: %s", pmid, exc)
        return DocumentMap(pmid=pmid, source_type="user_xml",
                           chunks=[], arms=[], timepoints=[])

    chunks: List[TextChunk] = []
    counters: dict = {}

    def _chunk_id(section: str, ctype: str) -> str:
        key = f"{section}_{ctype}"
        counters[key] = counters.get(key, 0) + 1
        return f"{key}_{counters[key]}"

    # ── Abstract ─────────────────────────────────────────────────────────────
    for abstract in root.iter("abstract"):
        for p in abstract.iter("p"):
            text = "".join(p.itertext()).strip()
            if text:
                chunks.append(TextChunk(
                    chunk_id=_chunk_id("Abstract", "Para"),
                    section="Abstract",
                    content=text,
                    chunk_type="paragraph",
                ))

    # ── Body sections ─────────────────────────────────────────────────────────
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        section_label = _detect_section(
            "".join(title_el.itertext()) if title_el is not None else ""
        )

        # Paragraphs
        for p in sec.findall("p"):
            text = "".join(p.itertext()).strip()
            if text:
                chunks.append(TextChunk(
                    chunk_id=_chunk_id(section_label, "Para"),
                    section=section_label,
                    content=text,
                    chunk_type="paragraph",
                ))

        # Tables
        for tw in sec.findall(".//table-wrap"):
            rows: List[str] = []
            for tr in tw.iter("tr"):
                cells = [
                    "".join(td.itertext()).strip()
                    for td in tr
                    if td.tag in ("td", "th")
                ]
                rows.append(" | ".join(cells))
            table_text = "\n".join(rows)

            caption_el = tw.find(".//caption")
            caption = (
                "".join(caption_el.itertext()).strip()
                if caption_el is not None else ""
            )
            content = f"{caption}\n{table_text}".strip() if caption else table_text

            if content:
                chunks.append(TextChunk(
                    chunk_id=_chunk_id(section_label, "Table"),
                    section=section_label,
                    content=content,
                    chunk_type="table",
                ))

        # Figure captions
        for fig in sec.findall(".//fig"):
            caption_el = fig.find(".//caption")
            if caption_el is not None:
                text = "".join(caption_el.itertext()).strip()
                if text:
                    chunks.append(TextChunk(
                        chunk_id=_chunk_id(section_label, "FigCaption"),
                        section=section_label,
                        content=text,
                        chunk_type="figure_caption",
                    ))

    logger.info(
        "[TextParser] XML parsed: pmid=%s, %d chunks", pmid, len(chunks)
    )
    return DocumentMap(
        pmid=pmid,
        source_type="user_xml",
        chunks=chunks,
        arms=[],         # populated by Node 3.1 Soft sub-step
        timepoints=[],   # populated by Node 3.1 Soft sub-step
    )


# ─────────────────────────────────────────────────────────────────────────────
# PDF parser (fallback path — vision LLM)
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf(
    pmid: str,
    pdf_path: str,
    vision_agent: "ExecutorAgent",
) -> DocumentMap:
    """
    Parse a PDF by rendering each page to an image and sending to a
    vision-capable LLM to extract structured text.

    Requires: pymupdf (fitz)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error(
            "[TextParser] pymupdf not installed. Cannot parse PDF for pmid=%s", pmid
        )
        return DocumentMap(pmid=pmid, source_type="user_pdf",
                           chunks=[], arms=[], timepoints=[])

    import tempfile
    import base64

    doc = fitz.open(pdf_path)
    chunks: List[TextChunk] = []
    counters: dict = {}

    def _chunk_id(section: str, ctype: str) -> str:
        key = f"{section}_{ctype}"
        counters[key] = counters.get(key, 0) + 1
        return f"{key}_{counters[key]}"

    for page_num, page in enumerate(doc, start=1):
        # Render page to PNG
        pix = page.get_pixmap(dpi=150)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pix.save(f.name)
            tmp_path = f.name

        # Vision LLM call — structured text extraction
        # The agent is expected to be a vision-capable ExecutorAgent.
        # We pass a pseudo-state with the image path; the real implementation
        # would encode the image and call the LLM with a multimodal prompt.
        from src.engine.context_manager import MountedContext
        mounted = MountedContext(
            system_message=(
                "Extract all text from this page. Preserve table structure "
                "using Markdown table format. Identify the section (Methods, "
                "Results, etc.) based on headings."
            ),
            user_message=f"[Page {page_num} image at: {tmp_path}]",
            model_id=vision_agent.model_id,
            temperature=0.0,
            response_format="text",
        )
        raw_text = vision_agent.call(mounted)
        os.unlink(tmp_path)

        # Parse the LLM's structured output into TextChunks
        section = "Results"   # default; a real implementation would parse headings
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        for para in paragraphs:
            is_table = para.startswith("|") or "|" in para[:50]
            ctype = "table" if is_table else "paragraph"
            chunk_type_lit = "table" if is_table else "paragraph"
            chunks.append(TextChunk(
                chunk_id=_chunk_id(section, "Table" if is_table else "Para"),
                section=section,
                content=para,
                chunk_type=chunk_type_lit,
            ))

    doc.close()
    logger.info(
        "[TextParser] PDF parsed via vision LLM: pmid=%s, %d chunks",
        pmid, len(chunks),
    )
    return DocumentMap(
        pmid=pmid,
        source_type="user_pdf",
        chunks=chunks,
        arms=[],
        timepoints=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher — automatically picks XML or PDF
# ─────────────────────────────────────────────────────────────────────────────

def parse_document(
    pmid: str,
    file_path: str,
    vision_agent: Optional["ExecutorAgent"] = None,
) -> DocumentMap:
    """
    Dispatch to XML or PDF parser based on file extension.

    The resulting DocumentMap is format-agnostic — downstream nodes never
    need to know whether input was XML or PDF.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xml",):
        return parse_xml(pmid, file_path)
    elif ext in (".pdf",):
        if vision_agent is None:
            raise ValueError(
                f"PDF parsing requires a vision-capable agent for pmid={pmid}"
            )
        return parse_pdf(pmid, file_path, vision_agent)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}' for pmid={pmid}. "
            "Only .xml and .pdf are supported."
        )


def first_sentence(text: str) -> str:
    """Extract the first sentence from a text chunk (for Node 3.3a index)."""
    text = text.strip()
    # Split on sentence-ending punctuation followed by whitespace or EOL
    match = re.search(r"[.!?][\s\n]", text)
    if match:
        return text[: match.start() + 1].strip()
    return text[:200].strip()   # fallback: first 200 chars
