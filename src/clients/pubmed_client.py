"""
Async PubMed E-utilities client.

Spec: docs/05_SEARCH_STAGE.md §3

All NCBI API calls go through this client with:
- AsyncRateLimiter (token-bucket): 10 req/s with API key, 3 req/s without.
- Retry on HTTP 429 (back off 1 s) and HTTP 500/503 (exponential backoff x3).
- Uses aiohttp.ClientSession for all HTTP.
- XML parsing via lxml.etree.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlencode

import aiohttp
from lxml import etree
from pydantic import BaseModel

from src.schemas.common import PaperMetadata

logger = logging.getLogger("autosr.pubmed_client")

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# ─────────────────────────────────────────────────────────────────────────────
# Internal result types
# ─────────────────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    webenv: str
    query_key: str
    total_count: int
    pmid_list: List[str]


class MeSHResult(BaseModel):
    found: bool
    descriptor_name: Optional[str] = None
    entry_terms: List[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# AsyncRateLimiter
# ─────────────────────────────────────────────────────────────────────────────

class AsyncRateLimiter:
    """
    Token-bucket rate limiter for async HTTP calls.

    - rate: max requests per second
    - Enforces minimum interval between acquisitions.
    - Callers `await limiter.acquire()` before every request.
    """

    def __init__(self, rate: int) -> None:
        self._semaphore = asyncio.Semaphore(rate)
        self._min_interval = 1.0 / rate
        self._last_call: float = 0.0

    async def acquire(self) -> None:
        async with self._semaphore:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = asyncio.get_event_loop().time()


# ─────────────────────────────────────────────────────────────────────────────
# PubMedClient
# ─────────────────────────────────────────────────────────────────────────────

class PubMedClient:
    """
    Async wrapper for NCBI E-utilities (esearch, efetch, espell, MeSH lookup).

    Usage:
        client = PubMedClient(api_key=os.environ.get("NCBI_API_KEY"))
        async with client:
            result = await client.search("diabetes[MeSH Terms]")
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("NCBI_API_KEY")
        rate = 10 if self._api_key else 3
        self._limiter = AsyncRateLimiter(rate)
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Context manager ──────────────────────────────────────────────────────

    async def __aenter__(self) -> "PubMedClient":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Core HTTP helper ─────────────────────────────────────────────────────

    async def _get(self, endpoint: str, params: dict) -> str:
        """
        GET request to NCBI E-utilities.

        Retries:
        - HTTP 429 → sleep 1 s, retry (up to 3 times)
        - HTTP 500/503 → exponential backoff (1 s, 2 s, 4 s)
        """
        if self._api_key:
            params["api_key"] = self._api_key

        url = _BASE_URL + endpoint
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            await self._limiter.acquire()
            try:
                async with self._session.get(url, params=params,
                                              timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    elif resp.status == 429:
                        wait = 1.0
                        logger.warning("[PubMed] 429 rate-limit hit. Sleeping %ss.", wait)
                        await asyncio.sleep(wait)
                    elif resp.status in (500, 503):
                        wait = 2 ** (attempt - 1)
                        logger.warning("[PubMed] HTTP %d. Backing off %ss (attempt %d).",
                                       resp.status, wait, attempt)
                        await asyncio.sleep(wait)
                    else:
                        text = await resp.text()
                        raise RuntimeError(
                            f"NCBI HTTP {resp.status} for {endpoint}: {text[:200]}"
                        )
            except aiohttp.ClientError as exc:
                if attempt == max_attempts:
                    raise
                logger.warning("[PubMed] Request error attempt %d: %s", attempt, exc)
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"NCBI request failed after {max_attempts} attempts: {url}")

    # ── Public API ───────────────────────────────────────────────────────────

    async def search(self, query: str, use_history: bool = True) -> SearchResult:
        """Execute esearch; always uses History Server (usehistory=y)."""
        params = {
            "db": "pubmed",
            "term": query,
            "usehistory": "y",
            "retmode": "json",
            "retmax": "0",   # We only want metadata, not PMIDs yet
        }
        raw = await self._get("esearch.fcgi", params)
        import json
        data = json.loads(raw)["esearchresult"]
        return SearchResult(
            webenv=data["webenv"],
            query_key=data["querykey"],
            total_count=int(data["count"]),
            pmid_list=data.get("idlist", []),
        )

    async def fetch_batch(
        self,
        webenv: str,
        query_key: str,
        retstart: int,
        retmax: int = 500,
    ) -> List[PaperMetadata]:
        """Fetch a single page of results from the History Server and parse XML."""
        params = {
            "db": "pubmed",
            "query_key": query_key,
            "WebEnv": webenv,
            "retstart": str(retstart),
            "retmax": str(retmax),
            "rettype": "xml",
            "retmode": "xml",
        }
        raw = await self._get("efetch.fcgi", params)
        return self._parse_pubmed_xml(raw)

    async def fetch_all(
        self,
        webenv: str,
        query_key: str,
        total_count: int,
        retmax: int = 500,
    ) -> List[PaperMetadata]:
        """
        Paginated batch fetch of all records.
        Concurrent requests controlled by the rate limiter's semaphore.
        """
        tasks = [
            self.fetch_batch(webenv, query_key, retstart=i, retmax=retmax)
            for i in range(0, total_count, retmax)
        ]
        pages = await asyncio.gather(*tasks)
        results: List[PaperMetadata] = []
        for page in pages:
            results.extend(page)
        return results

    async def validate_mesh(self, term: str) -> MeSHResult:
        """
        Query the NCBI MeSH database for a single term.
        Returns found=True if the term matches a MeSH descriptor or entry term.
        """
        # Step 1: esearch in mesh db to find the descriptor record
        params_search = {
            "db": "mesh",
            "term": f'"{term}"[Full Word]',
            "retmax": "1",
            "retmode": "json",
        }
        raw_search = await self._get("esearch.fcgi", params_search)
        import json
        data = json.loads(raw_search)["esearchresult"]
        ids = data.get("idlist", [])

        if not ids:
            # Also try without full-word constraint
            params_search2 = {
                "db": "mesh",
                "term": term,
                "retmax": "1",
                "retmode": "json",
            }
            raw2 = await self._get("esearch.fcgi", params_search2)
            data2 = json.loads(raw2)["esearchresult"]
            ids = data2.get("idlist", [])

        if not ids:
            return MeSHResult(found=False)

        # Step 2: efetch the MeSH record
        params_fetch = {
            "db": "mesh",
            "id": ids[0],
            "rettype": "full",
            "retmode": "xml",
        }
        raw_mesh = await self._get("efetch.fcgi", params_fetch)
        return self._parse_mesh_xml(term, raw_mesh)

    async def spell_check(self, term: str) -> Optional[str]:
        """Use NCBI espell to get a spelling suggestion. Returns None if no suggestion."""
        params = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
        }
        raw = await self._get("espell.fcgi", params)
        import json
        data = json.loads(raw).get("espellresult", {})
        corrected = data.get("correctedquery", "").strip()
        if corrected and corrected.lower() != term.lower():
            return corrected
        return None

    # ── XML Parsers ──────────────────────────────────────────────────────────

    def _parse_pubmed_xml(self, xml_text: str) -> List[PaperMetadata]:
        """Parse PubMed efetch XML into a list of PaperMetadata."""
        results: List[PaperMetadata] = []
        try:
            root = etree.fromstring(xml_text.encode("utf-8"))
        except etree.XMLSyntaxError as exc:
            logger.error("[PubMed] XML parse error: %s", exc)
            return results

        for article in root.iter("PubmedArticle"):
            try:
                meta = self._extract_paper_metadata(article)
                if meta:
                    results.append(meta)
            except Exception as exc:
                logger.warning("[PubMed] Failed to parse article: %s", exc)
        return results

    def _extract_paper_metadata(self, article_elem) -> Optional[PaperMetadata]:
        """Extract PaperMetadata fields from a single <PubmedArticle> element."""
        # PMID
        pmid_el = article_elem.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            return None
        pmid = pmid_el.text.strip()

        # Title
        title_el = article_elem.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstract
        abstract_parts = []
        for text_el in article_elem.findall(".//AbstractText"):
            label = text_el.get("Label", "")
            text = "".join(text_el.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Publication types
        pub_types = [
            pt.text.strip()
            for pt in article_elem.findall(".//PublicationType")
            if pt.text
        ]

        # MeSH terms
        mesh_terms = [
            mh.find("DescriptorName").text.strip()
            for mh in article_elem.findall(".//MeshHeading")
            if mh.find("DescriptorName") is not None
        ]

        return PaperMetadata(
            pmid=pmid,
            title=title,
            abstract=abstract,
            publication_types=pub_types,
            mesh_terms=mesh_terms,
            fetch_date=datetime.now(timezone.utc),
        )

    def _parse_mesh_xml(self, query_term: str, xml_text: str) -> MeSHResult:
        """Parse NCBI MeSH efetch XML to extract descriptor name and entry terms."""
        try:
            root = etree.fromstring(xml_text.encode("utf-8"))
        except etree.XMLSyntaxError:
            return MeSHResult(found=False)

        # The MeSH XML structure: <DescriptorRecord> → <DescriptorName> + <ConceptList>
        descriptor_el = root.find(".//DescriptorName/String")
        if descriptor_el is None:
            return MeSHResult(found=False)

        descriptor_name = descriptor_el.text.strip() if descriptor_el.text else None
        if not descriptor_name:
            return MeSHResult(found=False)

        # Entry terms: all <Term> elements in <ConceptList>
        entry_terms = [
            t.text.strip()
            for t in root.findall(".//Term/String")
            if t.text and t.text.strip()
        ]

        # Determine `found`: True if query_term matches the descriptor or any entry term
        query_lower = query_term.lower()
        all_names = [descriptor_name] + entry_terms
        found = any(name.lower() == query_lower for name in all_names)

        return MeSHResult(
            found=found,
            descriptor_name=descriptor_name,
            entry_terms=entry_terms,
        )
