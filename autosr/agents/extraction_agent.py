"""
ExtractionAgent — extracts structured data from uploaded PDFs.

Four-step pipeline:
  Step 1. Parse PDFs using Docling              (pdf_parser.parse_pdfs)
  Step 2. Chunk text + BM25 semantic filtering   (chunker.chunk_document + build_context)
  Step 3. LLM extraction with GPT-5.4           (batch_function_call_llm with model override)
     3a. Study characteristics (one row per paper)
     3b. Study results (multiple rows per paper)
  Step 4. Assemble output tables                 (deterministic aggregation)
"""

import logging
from typing import Generator, List, Tuple

from autosr.agents.base_agent import BaseAgent
from autosr.schemas.models import PICODefinition
from autosr.schemas.extraction_models import (
    ExtractionFieldDefinition,
    ParsedPDF,
    FieldExtraction,
    CharacteristicsRow,
    ResultsRow,
    ExtractionOutput,
    ExtractionSummary,
)
from autosr.tools.llm import batch_function_call_llm
from autosr.tools.pdf_parser import parse_pdfs, get_last_ocr_fallback_files
from autosr.tools.chunker import (
    chunk_document,
    build_context_chunks,
    format_chunks_with_citations,
)
from autosr.prompts.extraction import (
    STUDY_CHARACTERISTICS_EXTRACTION,
    STUDY_RESULTS_EXTRACTION,
)
from configs.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema builders
# ---------------------------------------------------------------------------

def _build_characteristics_tool(field_names: List[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "submit_characteristics",
            "description": (
                f"Submit extracted study characteristics. "
                f"Return exactly {len(field_names)} field extractions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "extractions": {
                        "type": "array",
                        "description": f"Exactly {len(field_names)} field extractions, one per field.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field_name": {"type": "string"},
                                "value": {"type": "string", "description": "Extracted value or 'NOT FOUND'"},
                                "citation": {"type": "string", "description": "Verbatim quote from the paper"},
                                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                            },
                            "required": ["field_name", "value", "confidence"],
                        },
                        "minItems": len(field_names),
                        "maxItems": len(field_names),
                    }
                },
                "required": ["extractions"],
            },
        },
    }


def _build_results_tool(field_names: List[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "submit_results",
            "description": (
                "Submit extracted study results. "
                "One or more rows per paper (one per outcome/subgroup/timepoint)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "description": "One row per distinct outcome/subgroup/timepoint.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "outcome_label": {
                                    "type": "string",
                                    "description": "Descriptive label, e.g. 'Primary: BMI at 6 months'",
                                },
                                "extractions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "field_name": {"type": "string"},
                                            "value": {"type": "string"},
                                            "citation": {"type": "string"},
                                            "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                                        },
                                        "required": ["field_name", "value", "confidence"],
                                    },
                                },
                            },
                            "required": ["outcome_label", "extractions"],
                        },
                        "minItems": 1,
                    }
                },
                "required": ["rows"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Helper: format field definitions for the prompt
# ---------------------------------------------------------------------------

def _format_fields_text(fields: List[ExtractionFieldDefinition]) -> str:
    lines = []
    for i, f in enumerate(fields, start=1):
        if f.description:
            lines.append(f"{i}. {f.name} — {f.description}")
        else:
            lines.append(f"{i}. {f.name}")
    return "\n".join(lines)


def _parse_model_chain(model_config: str) -> List[str]:
    """
    Parse model chain from a config string.
    Supports comma-separated list, e.g.:
      "anthropic/claude-sonnet-4.6, qwen/qwen3.6-plus"
    """
    if not model_config:
        return []
    models = [part.strip() for part in model_config.split(",")]
    models = [model for model in models if model]
    return models


# ---------------------------------------------------------------------------
# ExtractionAgent
# ---------------------------------------------------------------------------

class ExtractionAgent(BaseAgent):
    """
    Extracts structured study data from uploaded PDFs using Docling + BM25 + LLM.

    Usage::

        agent = ExtractionAgent()
        result = agent.run(
            file_paths=["paper1.pdf", "paper2.pdf"],
            pico=PICODefinition(P="...", I="...", C="...", O="..."),
            char_fields=[ExtractionFieldDefinition(name="Author"), ...],
            result_fields=[ExtractionFieldDefinition(name="Effect Size"), ...],
        )
    """

    def __init__(self):
        super().__init__("ExtractionAgent")
        self._model_chain = _parse_model_chain(settings.extraction_model_name)
        if not self._model_chain:
            self._model_chain = [settings.model_name]
        self._model = self._model_chain[0]
        logger.info(
            "[ExtractionAgent] Extraction model chain: %s",
            " -> ".join(self._model_chain),
        )

    def _batch_function_call_with_fallback(
        self,
        prompt_template: str,
        batch_inputs: list,
        tool: dict,
        max_concurrency: int,
        phase: str,
    ) -> List[dict]:
        """
        Call batch function-calling with model fallback.
        Tries models in self._model_chain order until one succeeds.
        """
        last_exc = None
        total = len(self._model_chain)

        for i, model_name in enumerate(self._model_chain, start=1):
            try:
                logger.info(
                    "[ExtractionAgent] %s using model: %s (%d/%d)",
                    phase, model_name, i, total,
                )
                outputs = batch_function_call_llm(
                    prompt_template,
                    batch_inputs,
                    tool=tool,
                    max_concurrency=max_concurrency,
                    model=model_name,
                )
                self._model = model_name
                logger.info(
                    "[ExtractionAgent] %s succeeded with model: %s",
                    phase, model_name,
                )
                return outputs
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[ExtractionAgent] %s failed with model %s (%d/%d): %s",
                    phase, model_name, i, total, exc,
                )

        if last_exc:
            raise last_exc
        return []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        file_paths: List[str],
        pico: PICODefinition,
        char_fields: List[ExtractionFieldDefinition],
        result_fields: List[ExtractionFieldDefinition],
        top_k: int = 15,
        max_concurrency: int = 10,
    ) -> ExtractionOutput:
        self.reset()

        if not file_paths:
            return ExtractionOutput()

        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}

        # Step 1: Parse PDFs
        parsed_docs = self._run_step(
            "parse_pdfs", self._parse_pdfs, file_paths,
        )

        # Step 2: Chunk + semantic filtering
        char_contexts, result_contexts = self._run_step(
            "chunk_and_retrieve",
            self._chunk_and_retrieve,
            parsed_docs, char_fields, result_fields, top_k,
        )

        # Step 3a: Extract characteristics
        characteristics = []
        if char_fields:
            characteristics = self._run_step(
                "extract_characteristics",
                self._extract_characteristics,
                parsed_docs, char_contexts, pico_dict, char_fields, max_concurrency,
            )

        # Step 3b: Extract results
        results = []
        if result_fields:
            results = self._run_step(
                "extract_results",
                self._extract_results,
                parsed_docs, result_contexts, pico_dict, result_fields, max_concurrency,
            )

        output = ExtractionOutput(characteristics=characteristics, results=results)

        logger.info(
            "[ExtractionAgent] Done: %d papers → %d char rows, %d result rows (%.1fs)",
            len(file_paths), len(characteristics), len(results), self.state.elapsed,
        )
        return output

    # ------------------------------------------------------------------
    # Step 1: PDF Parsing
    # ------------------------------------------------------------------

    def _parse_pdfs(self, file_paths: List[str]) -> List[ParsedPDF]:
        raw_results = parse_pdfs(file_paths)
        docs = []
        for filename, md, tables, pages in raw_results:
            docs.append(ParsedPDF(
                filename=filename,
                markdown_text=md,
                tables=tables,
                num_pages=pages,
            ))
        logger.info(
            "[ExtractionAgent] Parsed %d/%d PDFs successfully",
            sum(1 for d in docs if d.markdown_text), len(file_paths),
        )
        return docs

    # ------------------------------------------------------------------
    # Step 2: Chunk + BM25 Retrieve
    # ------------------------------------------------------------------

    def _chunk_and_retrieve(
        self,
        docs: List[ParsedPDF],
        char_fields: List[ExtractionFieldDefinition],
        result_fields: List[ExtractionFieldDefinition],
        top_k: int,
    ) -> Tuple[List[str], List[str]]:
        """
        For each document, chunk body text and retrieve relevant context.

        Returns:
            (char_contexts, result_contexts) — each is a list of formatted
            chunks_text strings, one per document.
        """
        char_field_tuples = [(f.name, f.description) for f in char_fields]
        result_field_tuples = [(f.name, f.description) for f in result_fields]

        char_contexts = []
        result_contexts = []

        for doc in docs:
            if not doc.markdown_text:
                char_contexts.append("(PDF parsing failed — no content available)")
                result_contexts.append("(PDF parsing failed — no content available)")
                continue

            body_chunks, table_chunks = chunk_document(
                doc.markdown_text, doc.tables,
            )

            # Characteristics context
            if char_field_tuples:
                char_chunks = build_context_chunks(
                    body_chunks, table_chunks, char_field_tuples, top_k=top_k,
                )
                char_contexts.append(format_chunks_with_citations(char_chunks))
            else:
                char_contexts.append("")

            # Results context
            if result_field_tuples:
                result_chunks = build_context_chunks(
                    body_chunks, table_chunks, result_field_tuples, top_k=top_k,
                )
                result_contexts.append(format_chunks_with_citations(result_chunks))
            else:
                result_contexts.append("")

        logger.info(
            "[ExtractionAgent] Chunked %d documents, contexts ready", len(docs),
        )
        return char_contexts, result_contexts

    # ------------------------------------------------------------------
    # Step 3a: Extract Characteristics
    # ------------------------------------------------------------------

    def _extract_characteristics(
        self,
        docs: List[ParsedPDF],
        char_contexts: List[str],
        pico_dict: dict,
        char_fields: List[ExtractionFieldDefinition],
        max_concurrency: int,
    ) -> List[CharacteristicsRow]:
        field_names = [f.name for f in char_fields]
        fields_text = _format_fields_text(char_fields)
        tool = _build_characteristics_tool(field_names)

        batch_inputs = []
        valid_indices = []
        for i, doc in enumerate(docs):
            if not doc.markdown_text:
                continue
            batch_inputs.append({
                **pico_dict,
                "fields_text": fields_text,
                "chunks_text": char_contexts[i],
            })
            valid_indices.append(i)

        if not batch_inputs:
            return [CharacteristicsRow(filename=d.filename) for d in docs]

        raw_results = self._batch_function_call_with_fallback(
            prompt_template=STUDY_CHARACTERISTICS_EXTRACTION,
            batch_inputs=batch_inputs,
            tool=tool,
            max_concurrency=max_concurrency,
            phase="extract_characteristics",
        )

        # Map results back to all documents
        result_map = {}
        for idx, raw in zip(valid_indices, raw_results):
            result_map[idx] = raw

        rows = []
        for i, doc in enumerate(docs):
            raw = result_map.get(i, {})
            extractions = self._parse_characteristics(raw, field_names)
            rows.append(CharacteristicsRow(filename=doc.filename, extractions=extractions))

        return rows

    def _parse_characteristics(
        self, raw: dict, field_names: List[str],
    ) -> List[FieldExtraction]:
        raw_extractions = raw.get("extractions", [])

        # Build lookup by field_name
        lookup = {}
        for ext in raw_extractions:
            if isinstance(ext, dict):
                name = ext.get("field_name", "")
                lookup[name] = ext

        result = []
        for fn in field_names:
            ext = lookup.get(fn, {})
            result.append(FieldExtraction(
                field_name=fn,
                value=ext.get("value", "NOT FOUND"),
                citation=ext.get("citation", ""),
                confidence=ext.get("confidence", "LOW") if ext.get("value", "NOT FOUND") != "NOT FOUND" else "LOW",
            ))
        return result

    # ------------------------------------------------------------------
    # Step 3b: Extract Results
    # ------------------------------------------------------------------

    def _extract_results(
        self,
        docs: List[ParsedPDF],
        result_contexts: List[str],
        pico_dict: dict,
        result_fields: List[ExtractionFieldDefinition],
        max_concurrency: int,
    ) -> List[ResultsRow]:
        field_names = [f.name for f in result_fields]
        fields_text = _format_fields_text(result_fields)
        tool = _build_results_tool(field_names)

        batch_inputs = []
        valid_indices = []
        for i, doc in enumerate(docs):
            if not doc.markdown_text:
                continue
            batch_inputs.append({
                **pico_dict,
                "fields_text": fields_text,
                "chunks_text": result_contexts[i],
            })
            valid_indices.append(i)

        if not batch_inputs:
            return []

        raw_results = self._batch_function_call_with_fallback(
            prompt_template=STUDY_RESULTS_EXTRACTION,
            batch_inputs=batch_inputs,
            tool=tool,
            max_concurrency=max_concurrency,
            phase="extract_results",
        )

        all_rows = []
        for idx, raw in zip(valid_indices, raw_results):
            doc = docs[idx]
            rows = self._parse_results(raw, doc.filename, field_names)
            all_rows.extend(rows)

        return all_rows

    def _parse_results(
        self, raw: dict, filename: str, field_names: List[str],
    ) -> List[ResultsRow]:
        raw_rows = raw.get("rows", [])
        if not raw_rows:
            return []

        result_rows = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                continue
            outcome_label = raw_row.get("outcome_label", "")
            raw_extractions = raw_row.get("extractions", [])

            # Build lookup
            lookup = {}
            for ext in raw_extractions:
                if isinstance(ext, dict):
                    name = ext.get("field_name", "")
                    lookup[name] = ext

            extractions = []
            for fn in field_names:
                ext = lookup.get(fn, {})
                extractions.append(FieldExtraction(
                    field_name=fn,
                    value=ext.get("value", "NOT FOUND"),
                    citation=ext.get("citation", ""),
                    confidence=ext.get("confidence", "LOW") if ext.get("value", "NOT FOUND") != "NOT FOUND" else "LOW",
                ))

            result_rows.append(ResultsRow(
                filename=filename,
                outcome_label=outcome_label,
                extractions=extractions,
            ))

        return result_rows

    # ------------------------------------------------------------------
    # Streaming entry point (yields SSE-friendly dicts)
    # ------------------------------------------------------------------

    def run_stream(
        self,
        file_paths: List[str],
        pico: PICODefinition,
        char_fields: List[ExtractionFieldDefinition],
        result_fields: List[ExtractionFieldDefinition],
        top_k: int = 15,
        max_concurrency: int = 10,
    ) -> Generator[dict, None, None]:
        """
        Generator that yields progress events for SSE streaming.

        Event types:
          {"type": "parsing",            "data": {"filename": str, "status": "ok"/"failed"}}
          {"type": "parsing_done",       "data": {"total": N, "parsed": M}}
          {"type": "ocr_fallback",       "data": {"message": str, "files": [str, ...], "count": N}}
          {"type": "chunking_done",      "data": {"total_documents": N}}
          {"type": "extraction_start",   "data": {"kind": "characteristics"/"results"}}
          {"type": "paper_extracted",    "data": {"filename": str, "characteristics": {...}, "results": [...]}}
          {"type": "summary",            "data": ExtractionSummary}
          {"type": "done",               "data": ExtractionOutput}
          {"type": "error",              "data": str}
        """
        self.reset()

        if not file_paths:
            yield {"type": "done", "data": ExtractionOutput().model_dump()}
            return

        pico_dict = {"P": pico.P, "I": pico.I, "C": pico.C, "O": pico.O}

        # Step 1: Parse PDFs — yield per-file progress
        try:
            parsed_docs = []
            raw_results = parse_pdfs(file_paths)
            for filename, md, tables, pages in raw_results:
                doc = ParsedPDF(filename=filename, markdown_text=md, tables=tables, num_pages=pages)
                parsed_docs.append(doc)
                status = "ok" if md else "failed"
                yield {"type": "parsing", "data": {"filename": filename, "status": status}}

            parsed_count = sum(1 for d in parsed_docs if d.markdown_text)
            yield {"type": "parsing_done", "data": {"total": len(file_paths), "parsed": parsed_count}}

            fallback_files = get_last_ocr_fallback_files()
            if fallback_files:
                logger.warning(
                    "[ExtractionAgent] OCR fallback used for %d file(s): %s",
                    len(fallback_files),
                    ", ".join(fallback_files),
                )
                yield {
                    "type": "ocr_fallback",
                    "data": {
                        "message": "OCR fallback used",
                        "files": fallback_files,
                        "count": len(fallback_files),
                    },
                }
        except Exception as exc:
            logger.exception("run_stream: parse_pdfs failed")
            yield {"type": "error", "data": str(exc)}
            return

        # Step 2: Chunk + retrieve
        try:
            char_contexts, result_contexts = self._chunk_and_retrieve(
                parsed_docs, char_fields, result_fields, top_k,
            )
            yield {"type": "chunking_done", "data": {"total_documents": len(parsed_docs)}}
        except Exception as exc:
            logger.exception("run_stream: chunking failed")
            yield {"type": "error", "data": str(exc)}
            return

        # Step 3a: Extract characteristics
        characteristics = []
        if char_fields:
            try:
                yield {"type": "extraction_start", "data": {"kind": "characteristics"}}
                characteristics = self._extract_characteristics(
                    parsed_docs, char_contexts, pico_dict, char_fields, max_concurrency,
                )
            except Exception as exc:
                logger.exception("run_stream: characteristics extraction failed")
                yield {"type": "error", "data": str(exc)}
                return

        # Step 3b: Extract results
        results = []
        if result_fields:
            try:
                yield {"type": "extraction_start", "data": {"kind": "results"}}
                results = self._extract_results(
                    parsed_docs, result_contexts, pico_dict, result_fields, max_concurrency,
                )
            except Exception as exc:
                logger.exception("run_stream: results extraction failed")
                yield {"type": "error", "data": str(exc)}
                return

        # Yield per-paper combined results
        char_map = {row.filename: row for row in characteristics}
        results_map: dict = {}
        for row in results:
            results_map.setdefault(row.filename, []).append(row)

        for doc in parsed_docs:
            char_row = char_map.get(doc.filename)
            res_rows = results_map.get(doc.filename, [])
            yield {
                "type": "paper_extracted",
                "data": {
                    "filename": doc.filename,
                    "characteristics": char_row.model_dump() if char_row else None,
                    "results": [r.model_dump() for r in res_rows],
                },
            }

        # Summary
        summary = ExtractionSummary(
            total_papers=len(file_paths),
            papers_parsed=sum(1 for d in parsed_docs if d.markdown_text),
            papers_extracted=len(characteristics),
            total_characteristics_fields=len(char_fields),
            total_results_fields=len(result_fields),
        )
        yield {"type": "summary", "data": summary.model_dump()}

        output = ExtractionOutput(characteristics=characteristics, results=results)
        yield {"type": "done", "data": output.model_dump()}
