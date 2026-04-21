"""
PDF parsing tool using Docling for high-quality document conversion.
Converts PDF files to Markdown text with table extraction.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Pre-downloaded models path; set DOCLING_ARTIFACTS_PATH env var to use local models
DOCLING_ARTIFACTS_PATH = os.environ.get("DOCLING_ARTIFACTS_PATH")
DOCLING_DISABLE_OCR = os.environ.get("DOCLING_DISABLE_OCR", "").lower() in {
    "1", "true", "yes",
}

# If OCR initialization fails once in this process, disable OCR for subsequent files.
_OCR_FALLBACK_MODE = False
_LAST_OCR_FALLBACK_FILES: List[str] = []


def _convert_pdf(file_path: str, do_ocr: bool) -> Tuple[str, List[str], int]:
    """
    Convert a single PDF file using Docling.

    Returns:
        (markdown_text, tables_as_markdown_strings, num_pages)
    """
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = do_ocr
    if DOCLING_ARTIFACTS_PATH:
        pipeline_options.artifacts_path = DOCLING_ARTIFACTS_PATH

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    result = converter.convert(file_path)
    doc = result.document

    # Full document as Markdown
    markdown_text = doc.export_to_markdown()

    # Extract tables separately as Markdown strings
    tables = []
    for table in doc.tables:
        try:
            table_df = table.export_to_dataframe(doc=doc)
            tables.append(table_df.to_markdown(index=False))
        except Exception:
            try:
                table_html = table.export_to_html(doc=doc)
                tables.append(table_html)
            except Exception:
                logger.warning("Failed to export table from %s", file_path)

    num_pages = len(doc.pages) if doc.pages else 0

    return markdown_text, tables, num_pages


def parse_pdf(file_path: str) -> Tuple[str, List[str], int]:
    """
    Parse a single PDF file using Docling.

    Returns:
        (markdown_text, tables_as_markdown_strings, num_pages)
    """
    global _OCR_FALLBACK_MODE

    ocr_enabled = not (DOCLING_DISABLE_OCR or _OCR_FALLBACK_MODE)
    if not ocr_enabled:
        return _convert_pdf(file_path, do_ocr=False)

    try:
        return _convert_pdf(file_path, do_ocr=True)
    except Exception as first_error:
        logger.warning(
            "OCR-enabled parse failed for %s, retrying with OCR disabled: %s",
            file_path,
            first_error,
        )
        _OCR_FALLBACK_MODE = True
        _LAST_OCR_FALLBACK_FILES.append(Path(file_path).name)
        return _convert_pdf(file_path, do_ocr=False)


def parse_pdfs(file_paths: List[str]) -> List[Tuple[str, str, List[str], int]]:
    """
    Parse multiple PDF files.

    Returns:
        List of (filename, markdown_text, tables, num_pages)
    """
    global _LAST_OCR_FALLBACK_FILES
    _LAST_OCR_FALLBACK_FILES = []

    results = []
    for fp in file_paths:
        filename = Path(fp).name
        try:
            md, tables, pages = parse_pdf(fp)
            results.append((filename, md, tables, pages))
            logger.info(
                "Parsed %s: %d chars, %d tables, %d pages",
                filename, len(md), len(tables), pages,
            )
        except Exception as e:
            logger.error("Failed to parse %s: %s", filename, e)
            results.append((filename, "", [], 0))
    return results


def get_last_ocr_fallback_files() -> List[str]:
    """
    Return filenames that triggered OCR->no-OCR fallback in the latest parse_pdfs call.
    """
    return list(_LAST_OCR_FALLBACK_FILES)
