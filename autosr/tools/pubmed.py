"""
PubMed API tool layer for AutoSR.

Adapted directly from TrialMind's pubmed.py.
Changes:
  - Import api_key from our settings instead of os.environ directly
  - Use settings.pubmed_api_key throughout
  - Removed BioC/PMC helpers (not needed for title+abstract workflow)
  - Kept: ReqPubmedID, ReqPubmedFull, PubmedAPIWrapper, pmid2papers
"""

import copy
import json
import traceback
import urllib.parse
import xml.etree.ElementTree as ET
import logging
from typing import Optional

import pandas as pd
import requests
import tenacity
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from configs.settings import settings

logger = logging.getLogger(__name__)

PMID_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="
PUBMED_EFETCH_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="
DEFAULT_MAX_PAGE_SIZE = 100
BATCH_REQUEST_SIZE = 400


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_with_retry(url: str, max_retries: int = 5) -> requests.Response:
    retry_strategy = Retry(
        total=max_retries,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session.get(url)


# ---------------------------------------------------------------------------
# XML parsers (adapted from TrialMind)
# ---------------------------------------------------------------------------

def _parse_xml_recursively(element):
    child_dict = {}
    if element.text and element.text.strip():
        child_dict["text"] = element.text.strip()
    for child in element:
        if child.tag not in child_dict:
            child_dict[child.tag] = []
        child_dict[child.tag].append(_parse_xml_recursively(child))
    for key in list(child_dict.keys()):
        if isinstance(child_dict[key], list):
            if len(child_dict[key]) == 1:
                child_dict[key] = child_dict[key][0]
            elif len(child_dict[key]) == 0:
                del child_dict[key]
    return child_dict


def _parse_article_xml_to_dict(article) -> dict:
    results = {}
    d = _parse_xml_recursively(article)
    med = d.get("MedlineCitation", {})
    art = med.get("Article", {})

    results["PMID"] = med.get("PMID", {}).get("text", "")

    journal = art.get("Journal", {})
    results["Journal"] = journal.get("Title", {}).get("text", "")

    issue = journal.get("JournalIssue", {})
    pub_date = issue.get("PubDate", {})
    results["Year"] = pub_date.get("Year", {}).get("text", "")
    results["Month"] = pub_date.get("Month", {}).get("text", "")
    results["Day"] = pub_date.get("Day", {}).get("text", "")

    results["Title"] = art.get("ArticleTitle", {}).get("text", "")

    pub_types = art.get("PublicationTypeList", {}).get("PublicationType", [])
    if isinstance(pub_types, dict):
        pub_types = [pub_types]
    results["PublicationType"] = ", ".join(
        pt.get("text", "") if isinstance(pt, dict) else str(pt) for pt in pub_types
    )

    authors_raw = art.get("AuthorList", {}).get("Author", [])
    if isinstance(authors_raw, dict):
        authors_raw = [authors_raw]
    authors = []
    for a in authors_raw:
        last = a.get("LastName", {}).get("text", "") if isinstance(a, dict) else ""
        first = a.get("ForeName", {}).get("text", "") if isinstance(a, dict) else ""
        authors.append(f"{first} {last}".strip())
    results["Authors"] = ", ".join(authors)

    abstracts = art.get("Abstract", {}).get("AbstractText", [])
    if isinstance(abstracts, dict):
        abstracts = [abstracts]
    abstract_parts = []
    for ab in abstracts:
        if isinstance(ab, dict):
            abstract_parts.append(ab.get("text", ""))
        else:
            abstract_parts.append(str(ab))
    results["Abstract"] = "\n".join(abstract_parts)

    return results


def _parse_book_xml_to_dict(book) -> dict:
    results = {}
    d = _parse_xml_recursively(book)
    bd = d.get("BookDocument", {})

    results["PMID"] = bd.get("PMID", {}).get("text", "")
    results["Title"] = bd.get("Book", {}).get("BookTitle", {}).get("text", "")

    pub_date = bd.get("Book", {}).get("PubDate", {})
    results["Year"] = pub_date.get("Year", {}).get("text", "")
    results["Month"] = pub_date.get("Month", {}).get("text", "")
    results["Day"] = pub_date.get("Day", {}).get("text", "")
    results["Journal"] = ""
    results["Authors"] = ""
    results["PublicationType"] = bd.get("PublicationType", {}).get("text", "")

    abstracts = bd.get("Abstract", {}).get("AbstractText", [])
    if isinstance(abstracts, dict):
        abstracts = [abstracts]
    results["Abstract"] = "\n".join(
        ab.get("text", "") if isinstance(ab, dict) else str(ab) for ab in abstracts
    )
    return results


# ---------------------------------------------------------------------------
# Batch abstract retrieval (used for main paper fetch)
# ---------------------------------------------------------------------------

def _retrieve_abstracts(pmids: list, api_key: str = "") -> pd.DataFrame:
    all_records = []
    for i in range(0, len(pmids), BATCH_REQUEST_SIZE):
        batch = pmids[i : i + BATCH_REQUEST_SIZE]
        pmid_str = ",".join(batch)
        key_param = f"&api_key={api_key}" if api_key else ""
        url = f"{PUBMED_EFETCH_BASE_URL}{pmid_str}&retmode=xml{key_param}"
        logger.info("Fetching abstracts: %s", url)
        resp = _get_with_retry(url)
        if resp.status_code != 200:
            logger.warning("efetch returned %d for batch %d", resp.status_code, i)
            continue
        tree = ET.fromstring(resp.text)
        for art in tree.findall(".//PubmedArticle"):
            try:
                all_records.append(_parse_article_xml_to_dict(art))
            except Exception:
                logger.debug("Failed to parse article: %s", traceback.format_exc())
        for book in tree.findall(".//PubmedBookArticle"):
            try:
                all_records.append(_parse_book_xml_to_dict(book))
            except Exception:
                pass
    if not all_records:
        return pd.DataFrame(columns=["PMID", "Title", "Abstract", "Authors", "Year", "Journal", "PublicationType"])
    return pd.DataFrame.from_records(all_records)


def pmid2papers(pmid_list: list, api_key: str = "") -> pd.DataFrame:
    """Fetch full metadata for a list of PMIDs. Returns a DataFrame."""
    if not pmid_list:
        return pd.DataFrame()
    return _retrieve_abstracts(pmid_list, api_key)


# ---------------------------------------------------------------------------
# ReqPubmedID – search for PMIDs by keyword term
# ---------------------------------------------------------------------------

class ReqPubmedID:
    """Fetch PubMed article IDs by keyword search (esearch API)."""

    def run(self, term: str, field: str = "Title/Abstract", retmax: int = 100) -> list:
        api_key = settings.pubmed_api_key
        # Only attach [field] filter if field is set AND term is a single token (no boolean ops)
        # For combined AND/OR queries, skip the field filter to avoid malformed syntax
        has_boolean = any(op in term for op in [" AND ", " OR ", "+AND+", "+OR+"])
        term_with_field = term if has_boolean or not field else f"{term}[{field}]"
        params = {
            "db": "pubmed",
            "term": term_with_field,
            "retmax": retmax,
            "retmode": "xml",
        }
        if api_key:
            params["api_key"] = api_key
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
        try:
            resp = requests.get(url, headers={"User-Agent": "AutoSR/1.0"})
            soup = BeautifulSoup(resp.text, "xml")
            return [tag.text for tag in soup.select("IdList Id")]
        except Exception:
            logger.error("ReqPubmedID failed: %s", traceback.format_exc())
            return []


# ---------------------------------------------------------------------------
# ReqPubmedFull – fetch title + abstract for a small set of PMIDs
# (used for reference paper context in search term generation)
# ---------------------------------------------------------------------------

class ReqPubmedFull:
    """Fetch title + abstract for a small list of PMIDs (efetch API)."""

    def run(self, pmids: list) -> list:
        if not pmids:
            return []
        api_key = settings.pubmed_api_key
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if api_key:
            params["api_key"] = api_key
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(params)
        try:
            resp = requests.get(url, headers={"User-Agent": "AutoSR/1.0"})
            soup = BeautifulSoup(resp.text, "xml")
            records = []
            for art in soup.select("PubmedArticle"):
                title = art.find("ArticleTitle")
                abstract = " ".join(n.text for n in art.select("AbstractText"))
                pubmed_id = None
                for aid in art.select("ArticleId"):
                    if aid.get("IdType") == "pubmed":
                        pubmed_id = aid.text
                records.append({
                    "title": title.text if title else "",
                    "abstract": abstract,
                    "pubmed_id": pubmed_id,
                })
            return records
        except Exception:
            logger.error("ReqPubmedFull failed: %s", traceback.format_exc())
            return []


# ---------------------------------------------------------------------------
# PubmedAPIWrapper – builds boolean query and retrieves PMIDs
# (mirrors TrialMind's PubmedAPIWrapper, streamlined for our use case)
# ---------------------------------------------------------------------------

class PubmedAPIWrapper:
    """
    Builds a PubMed boolean query from a keyword_map and retrieves PMIDs.

    Expected input dict::

        {
            "keyword_map": {
                "population":     ["term1", "term2"],
                "intervention":   ["term3", "term4"],
                "outcome":        ["term5", "term6"],
            },
            "page_size": 1000,          # optional, default 1000
            "min_date":  "2000",        # optional
            "max_date":  "2024",        # optional
        }

    The query structure: within each group → OR; across groups → AND.
    """

    @tenacity.retry(
        wait=tenacity.wait_fixed(2),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def _get_response(self, url: str) -> requests.Response:
        return requests.get(url, headers={"User-Agent": "AutoSR/1.0"})

    def build_query_string(self, inputs: dict) -> str:
        d = copy.deepcopy(inputs)
        api_key = settings.pubmed_api_key
        page_size = d.get("page_size", 1000)

        kw_map = d.get("keyword_map", {})
        group_queries = []
        for group_terms in kw_map.values():
            if group_terms:
                inner = "+OR+".join(
                    urllib.parse.quote(t, safe="") for t in group_terms
                )
                group_queries.append(f"({inner})")

        query_body = "+AND+".join(group_queries) if group_queries else ""

        filters = ""
        if d.get("min_date"):
            filters += f"&mindate={d['min_date']}"
        if d.get("max_date"):
            filters += f"&maxdate={d['max_date']}"

        key_param = f"&api_key={api_key}" if api_key else ""
        url = f"{PMID_BASE_URL}{query_body}&retmax={page_size}&retmode=json{filters}{key_param}"
        return url

    def search(self, inputs: dict):
        """
        Returns (pmid_list, query_url, total_count).
        pmid_list may be capped at page_size (default 1000).
        """
        url = self.build_query_string(inputs)
        logger.info("PubMed search URL: %s", url)
        try:
            resp = self._get_response(url)
            if resp.status_code != 200:
                logger.error("PubMed search error: %s", resp.text)
                return [], url, 0
            data = json.loads(resp.text, strict=False)
            pmid_list = data["esearchresult"]["idlist"]
            total_count = int(data["esearchresult"]["count"])
            pmid_list = list(set(pmid_list))
            logger.info("Retrieved %d PMIDs (total in PubMed: %d)", len(pmid_list), total_count)
            return pmid_list, url, total_count
        except Exception:
            logger.error("PubMed search failed: %s", traceback.format_exc())
            return [], url, 0

    def search_all(self, inputs: dict):
        """
        Fetch ALL PMIDs matching the query using retstart pagination.
        Ignores page_size in inputs; uses PAGE_SIZE=10000 per request.
        Returns (pmid_list, query_url, total_count).

        Note: for very large result sets (>50k) this may take a while —
        each page is a separate HTTP request.
        """
        PAGE_SIZE = 10_000

        # Build the base paginated URL (retmax=PAGE_SIZE)
        paged_inputs = dict(inputs)
        paged_inputs["page_size"] = PAGE_SIZE
        base_url = self.build_query_string(paged_inputs)
        query_url = base_url  # returned for display

        try:
            # First page — also gives us total_count
            first_url = base_url + "&retstart=0"
            logger.info("PubMed search_all (page 0): %s", first_url)
            resp = self._get_response(first_url)
            if resp.status_code != 200:
                logger.error("PubMed search_all error: %s", resp.text)
                return [], query_url, 0
            data = json.loads(resp.text, strict=False)
            total_count = int(data["esearchresult"]["count"])
            all_pmids: list = list(data["esearchresult"]["idlist"])
            logger.info(
                "search_all: total=%d, fetched so far=%d",
                total_count, len(all_pmids),
            )

            # Subsequent pages
            for retstart in range(PAGE_SIZE, total_count, PAGE_SIZE):
                page_url = base_url + f"&retstart={retstart}"
                logger.info("PubMed search_all (retstart=%d): %s", retstart, page_url)
                try:
                    resp = self._get_response(page_url)
                    if resp.status_code != 200:
                        logger.warning(
                            "search_all: page retstart=%d returned %d, stopping early",
                            retstart, resp.status_code,
                        )
                        break
                    page_data = json.loads(resp.text, strict=False)
                    batch = page_data["esearchresult"]["idlist"]
                    all_pmids.extend(batch)
                    logger.info(
                        "search_all: fetched %d/%d PMIDs (retstart=%d)",
                        len(all_pmids), total_count, retstart,
                    )
                except (json.JSONDecodeError, KeyError) as page_err:
                    logger.warning(
                        "search_all: page retstart=%d parse error (%s), skipping page",
                        retstart, page_err,
                    )
                    continue

            all_pmids = list(set(all_pmids))  # deduplicate
            logger.info(
                "search_all complete: %d unique PMIDs (PubMed total: %d)",
                len(all_pmids), total_count,
            )
            return all_pmids, query_url, total_count

        except Exception:
            logger.error("PubMed search_all failed: %s", traceback.format_exc())
            return all_pmids if 'all_pmids' in dir() else [], query_url, 0
