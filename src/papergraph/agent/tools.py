"""The agent's tools: what it can do in the world, as LangChain tools.

Each tool is a Python function decorated with LangChain's `@tool`: its name,
docstring and typed arguments become the JSON schema the model reads (no
hand-written schemas). `response_format="content_and_artifact"` makes each tool
return two things: compact text for the model (every token counts in a small
model's context) and the structured paper records, which travel on the
ToolMessage as its `artifact` for the UI and the citation check.
"""

import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx
import psycopg
from langchain_core.tools import BaseTool, tool

from papergraph import search as svc
from papergraph.retrieval.filters import Filters
from papergraph.search import Paper

ABSTRACT_CHARS = 350  # per paper in tool results; get_paper returns the full abstract


def format_paper(p: Paper, full: bool = False) -> str:
    year = (p.published or "")[:4]
    venue = f", {p.venue} {p.venue_year or ''}".rstrip() if p.venue else ""
    authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
    abstract = p.abstract if full else p.abstract[:ABSTRACT_CHARS]
    if not full and len(p.abstract) > ABSTRACT_CHARS:
        abstract += "..."
    return f"[{p.id}] {p.title} ({year}{venue}) — {authors}\n    {abstract}"


def format_papers(papers: list[Paper]) -> str:
    return "\n".join(format_paper(p) for p in papers) if papers else "No papers found."


def clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def as_list(value: Any) -> tuple[str, ...]:
    """Models send lists, comma-separated strings or single strings: accept all."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        value = value.split(",")
    return tuple(str(v).strip() for v in value if str(v).strip())


# --- arXiv live search ----------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_MIN_INTERVAL_S = 3.0  # arXiv asks API clients for at most one request every 3 s
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_last_arxiv_call = 0.0


# Words in nearly every ML abstract: requiring them (AND) adds no precision,
# and with results sorted by date it lets loosely related papers through.
GENERIC = frozenset(
    {
        "the", "and", "for", "with", "via", "from", "using", "based", "towards", "toward",
        "into", "large", "language", "model", "models", "learning", "deep", "neural",
        "network", "networks", "approach", "method", "methods", "paper", "study", "new",
    }
)  # fmt: skip


def arxiv_query(text: str) -> str:
    """'speculative decoding for large language models' -> 'all:speculative AND all:decoding'.

    Distinctive words only (generic ones are dropped unless nothing else is
    left), at most six, all required.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text) if len(w) > 2]
    distinctive = [w for w in words if w.lower() not in GENERIC] or words
    return " AND ".join(f"all:{w}" for w in distinctive[:6])


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    papers = []
    for entry in ET.fromstring(xml_text).findall("a:entry", _ATOM):
        raw_id = (entry.findtext("a:id", "", _ATOM) or "").rsplit("/abs/", 1)[-1]
        paper_id = re.sub(r"v\d+$", "", raw_id)  # 2511.01234v2 -> 2511.01234
        papers.append(
            Paper(
                id=paper_id,
                title=" ".join((entry.findtext("a:title", "", _ATOM) or "").split()),
                abstract=" ".join((entry.findtext("a:summary", "", _ATOM) or "").split()),
                authors=[
                    a.findtext("a:name", "", _ATOM) or "" for a in entry.findall("a:author", _ATOM)
                ],
                published=(entry.findtext("a:published", "", _ATOM) or "")[:10] or None,
                venue=None,
                venue_year=None,
                categories=[c.get("term", "") for c in entry.findall("a:category", _ATOM)],
            )
        )
    return papers


def fetch_arxiv(query: str, max_results: int, http: httpx.Client) -> list[Paper]:
    global _last_arxiv_call
    wait = ARXIV_MIN_INTERVAL_S - (time.monotonic() - _last_arxiv_call)
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_call = time.monotonic()
    resp = http.get(
        ARXIV_API,
        params={
            "search_query": arxiv_query(query),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        },
    )
    resp.raise_for_status()
    return parse_arxiv_feed(resp.text)


# --- the tools ----------------------------------------------------------------------

Result = tuple[str, list[Paper]]  # (text for the model, papers for the UI)


def make_tools(conn: psycopg.Connection, http: httpx.Client | None = None) -> list[BaseTool]:
    """The agent's tools, bound to a database connection (and an HTTP client for arXiv).

    Errors raised inside a tool, unknown tool names and invalid arguments are
    turned into messages the model can read by the agent's ToolErrorMiddleware
    (papergraph.agent.loop), so the tools themselves just raise.
    """
    client = http or httpx.Client(
        timeout=20, headers={"User-Agent": "PaperGraph research agent (educational project)"}
    )

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def search_papers(
        query: str,
        year_from: int | None = None,
        year_to: int | None = None,
        venues: list[str] | str | None = None,
        k: int = 8,
    ) -> Result:
        """Search the local index of ~581k arXiv ML papers (up to late 2026). Main tool. Use short, specific queries.

        Args:
            query: short search query
            year_from: earliest year
            year_to: latest year
            venues: only these venues, e.g. ['ICLR', 'NeurIPS']
            k: number of results (max 15)
        """  # noqa: E501
        filters = Filters(
            year_from=clamp(year_from, 1990, 2100, 0) or None,
            year_to=clamp(year_to, 1990, 2100, 0) or None,
            venues=as_list(venues),
        )
        papers = svc.search(conn, query, k=clamp(k, 1, 15, 8), filters=filters)
        return format_papers(papers), papers

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def papers_by_author(
        name: str,
        query: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        k: int = 10,
    ) -> Result:
        """Papers written by a given person (all spellings of the name). Use this for any question about a specific author; search_papers does NOT search author names. Optionally rank the author's papers by a topic.

        Args:
            name: e.g. 'Yann LeCun'
            query: optional topic to rank their papers by
            year_from: earliest year
            year_to: latest year
            k: number of results (max 20)
        """  # noqa: E501
        filters = Filters(
            year_from=clamp(year_from, 1990, 2100, 0) or None,
            year_to=clamp(year_to, 1990, 2100, 0) or None,
        )
        variants, papers = svc.author_papers(
            conn, name, query=query or None, k=clamp(k, 1, 20, 10), filters=filters
        )
        if not variants:
            return (
                f"No author matching {name!r} in the index. Check the spelling, or try the "
                "surname alone.",
                [],
            )
        names = ", ".join(f"{n} ({c} papers)" for n, c in variants)
        order = f"ranked by relevance to {query!r}" if query else "newest first"
        years = ""
        if filters.year_from or filters.year_to:
            years = f", published {filters.year_from or '...'}-{filters.year_to or 'now'}"
        header = (
            f"ALL {len(papers)} papers below are co-authored by {name} "
            f"(name spellings found: {names}){years}, {order}:"
        )
        return header + "\n" + format_papers(papers), papers

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def get_paper(paper_id: str) -> Result:
        """Get one paper's full abstract and metadata by its id.

        Args:
            paper_id: the paper's id, e.g. '2310.06825'
        """
        paper = svc.get_paper(conn, paper_id.strip("[] "))
        if paper is None:
            return f"No paper with id {paper_id!r} in the local index.", []
        return format_paper(paper, full=True), [paper]

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def similar_papers(paper_id: str, k: int = 8) -> Result:
        """Find papers most similar to a given paper (by embedding).

        Args:
            paper_id: the paper's id, e.g. '2310.06825'
            k: number of results (max 15)
        """
        papers = svc.similar_papers(conn, paper_id.strip("[] "), k=clamp(k, 1, 15, 8))
        if not papers:
            return f"No embedding for paper {paper_id!r}; try search_papers instead.", []
        return format_papers(papers), papers

    @tool(response_format="content_and_artifact", parse_docstring=True)
    def search_arxiv(query: str, max_results: int = 5) -> Result:
        """Live search of arXiv for the newest preprints, newest first. Slow; only for very recent work.

        Args:
            query: a few key terms
            max_results: max 10
        """  # noqa: E501
        papers = fetch_arxiv(query, clamp(max_results, 1, 10, 5), client)
        return format_papers(papers), papers

    return [search_papers, papers_by_author, get_paper, similar_papers, search_arxiv]
