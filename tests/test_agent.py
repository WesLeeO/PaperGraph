"""The research agent (LangChain), with a scripted fake chat model: no Ollama or database."""

from fakes import DownModel, FakeModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from papergraph.agent.loop import NUDGE, STEP_LIMIT, Event, cited_ids, run_agent
from papergraph.agent.tools import (
    arxiv_query,
    as_list,
    clamp,
    make_tools,
    parse_arxiv_feed,
)
from papergraph.search import Paper


def paper(pid: str) -> Paper:
    return Paper(
        pid, f"Title {pid}", "Abstract.", ["A. Author"], "2024-01-01", None, None, ["cs.CL"]
    )


_ids = iter(range(10_000))


def fake(*replies: AIMessage) -> FakeModel:
    """A chat model that returns `replies` in order, one per call."""
    return FakeModel(responses=list(replies))


def search(query: str = "rag hallucination") -> AIMessage:
    return AIMessage(
        "", tool_calls=[{"name": "search_papers", "args": {"query": query}, "id": f"c{next(_ids)}"}]
    )


def answer(text: str) -> AIMessage:
    return AIMessage(text)


def fake_tools(papers: list[Paper] | None = None, runs: list[str] | None = None) -> list[BaseTool]:
    found = papers if papers is not None else [paper("2401.00001"), paper("2401.00002")]

    @tool(response_format="content_and_artifact")
    def search_papers(query: str) -> tuple[str, list[Paper]]:
        """Search papers."""
        if runs is not None:
            runs.append(query)
        return "[2401.00001] Title ...", found

    @tool
    def broken(x: int) -> str:
        """Always fails."""
        raise RuntimeError("database is down")

    return [search_papers, broken]


def types(evs: list[Event]) -> list[str]:
    return [e.type for e in evs]


# --- the agent loop and its guardrails ------------------------------------------------


def test_tool_call_then_final_answer() -> None:
    runs: list[str] = []
    model = fake(search(), answer("Read [2401.00001] and [9999.99999]."))
    evs = list(run_agent("q", model, fake_tools(runs=runs), min_searches=1))
    assert types(evs) == ["tool_call", "tool_result", "final"]
    final = evs[-1].data
    # Only papers seen in tool results count; invented ids are flagged.
    assert [p["id"] for p in final["papers"]] == ["2401.00001"]
    assert final["unverified_ids"] == ["9999.99999"]
    assert final["steps"] == 1
    assert runs == ["rag hallucination"]
    assert evs[1].data["n_papers"] == 2 and evs[1].data["error"] is None


def test_answering_after_one_search_gets_one_nudge() -> None:
    runs: list[str] = []
    model = fake(search("a"), answer("too early"), search("b"), answer("ok [2401.00001]"))
    evs = list(run_agent("q", model, fake_tools(runs=runs)))
    assert types(evs).count("nudge") == 1
    assert runs == ["a", "b"]
    assert evs[-1].type == "final" and evs[-1].data["answer"] == "ok [2401.00001]"
    assert any(isinstance(m, HumanMessage) and m.content == NUDGE for m in model.seen[2])


def test_nudge_happens_only_once() -> None:
    model = fake(search("a"), answer("early"), answer("still early"))
    evs = list(run_agent("q", model, fake_tools()))
    assert types(evs).count("nudge") == 1
    assert evs[-1].data["answer"] == "still early"  # accepted the second time


def test_well_cited_answer_after_one_search_is_not_nudged() -> None:
    three = [paper("a.1"), paper("a.2"), paper("a.3")]
    model = fake(search(), answer("[a.1] [a.2] [a.3]"))
    evs = list(run_agent("q", model, fake_tools(three)))
    assert "nudge" not in types(evs)
    assert evs[-1].data["answer"] == "[a.1] [a.2] [a.3]"


def test_nudge_that_makes_the_answer_worse_is_undone() -> None:
    model = fake(search(), answer("Read [2401.00001]."), answer("I have fully answered."))
    evs = list(run_agent("q", model, fake_tools()))
    assert evs[-1].data["answer"] == "Read [2401.00001]."


def test_repeated_call_is_not_rerun() -> None:
    runs: list[str] = []
    model = fake(search("same"), search("same"), answer("done [2401.00001] [2401.00002] [x]"))
    evs = list(run_agent("q", model, fake_tools(runs=runs)))
    assert runs == ["same"]
    second = [e for e in evs if e.type == "tool_result"][1].data
    assert second["error"].startswith("You already")


def test_step_limit_forces_an_answer_without_tools() -> None:
    model = fake(search("a"), search("b"), search("c"), answer("final [2401.00002]"))
    evs = list(run_agent("q", model, fake_tools(), max_steps=3))
    assert evs[-1].type == "final" and evs[-1].data["steps"] == 3
    last_call = model.seen[-1]
    assert isinstance(last_call[-1], HumanMessage) and last_call[-1].content == STEP_LIMIT
    assert [p["id"] for p in evs[-1].data["papers"]] == ["2401.00002"]


def test_tool_errors_reach_the_model_instead_of_crashing() -> None:
    model = fake(
        AIMessage("", tool_calls=[{"name": "broken", "args": {"x": 1}, "id": "e1"}]),
        AIMessage("", tool_calls=[{"name": "delete_everything", "args": {}, "id": "e2"}]),
        AIMessage("", tool_calls=[{"name": "broken", "args": {"x": "lots"}, "id": "e3"}]),
        answer("sorry, no results"),
    )
    evs = list(run_agent("q", model, fake_tools(), min_searches=0))
    errors = [e.data["error"] for e in evs if e.type == "tool_result"]
    assert "database is down" in errors[0]  # a tool that raised
    assert "not a valid tool" in errors[1]  # a tool name the model invented
    assert "valid integer" in errors[2]  # a bad argument
    assert evs[-1].type == "final"


def test_unreachable_model_becomes_an_error_event() -> None:
    evs = list(run_agent("q", DownModel(), fake_tools()))
    assert types(evs) == ["error"]
    assert "cannot reach Ollama" in evs[0].data["message"]


def test_cited_ids() -> None:
    text = "See [2310.06825], [cs/0112017] and again [2310.06825]; not a [link text](x)."
    assert cited_ids(text) == ["2310.06825", "cs/0112017"]


# --- tools ------------------------------------------------------------------------------


def test_tools_have_names_and_schemas() -> None:
    tools = {t.name: t for t in make_tools(conn=None)}  # type: ignore[arg-type]
    assert set(tools) == {
        "search_papers",
        "papers_by_author",
        "get_paper",
        "similar_papers",
        "search_arxiv",
    }
    search_schema = tools["search_papers"].tool_call_schema.model_json_schema()
    assert search_schema["required"] == ["query"]
    assert "short search query" in search_schema["properties"]["query"]["description"]
    assert "does NOT search author names" in tools["papers_by_author"].description


def test_argument_coercion() -> None:
    assert clamp("12", 1, 15, 8) == 12
    assert clamp(99, 1, 15, 8) == 15
    assert clamp("lots", 1, 15, 8) == 8
    assert as_list("ICLR, NeurIPS") == ("ICLR", "NeurIPS")
    assert as_list(["ICML"]) == ("ICML",)
    assert as_list(None) == ()


def test_arxiv_query_keeps_distinctive_terms() -> None:
    assert arxiv_query("speculative decoding for large language models") == (
        "all:speculative AND all:decoding"
    )
    assert arxiv_query("sparse attention for long documents") == (
        "all:sparse AND all:attention AND all:long AND all:documents"
    )


def test_arxiv_query_falls_back_to_generic_words() -> None:
    assert arxiv_query("large language models") == "all:large AND all:language AND all:models"


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2511.01234v2</id>
    <published>2026-11-01T10:00:00Z</published>
    <title>A  New
      Method</title>
    <summary>We propose   a method.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <category term="cs.LG"/>
  </entry>
</feed>"""


def test_parse_arxiv_feed() -> None:
    (p,) = parse_arxiv_feed(ATOM)
    assert p.id == "2511.01234"  # version suffix stripped
    assert p.title == "A New Method"
    assert p.abstract == "We propose a method."
    assert p.authors == ["Ada Lovelace", "Alan Turing"]
    assert p.published == "2026-11-01"
    assert p.categories == ["cs.LG"]
    assert p.url == "https://arxiv.org/abs/2511.01234"
