"""The research agent: LangChain's `create_agent` plus PaperGraph's guardrails.

`create_agent` (built on LangGraph) runs the agent loop:

    model -> tool calls? -> run the tools -> model -> ... -> answer (no tool calls)

A small local model needs guardrails on top, added as LangChain middleware:

- ToolErrorMiddleware: a failing tool, an invented tool name or bad arguments
  become an error message the model can read and correct, not a crash.
- step_limit: after MAX_STEPS model calls, tools are withheld and the model
  is asked to answer with what it has (a forced final answer, not a cutoff).
- no_repeats: an exact repeat of an earlier tool call is not run again.
- nudge: a model that answers after one search with a thin answer is sent
  back once to cover the question's other aspects.
- ModelCallLimitMiddleware: a hard stop in case anything above misbehaves.

After the run, citations are checked: only papers that a tool actually
returned count (ids are taken from the tools' artifacts, not from the model).

`run_agent` translates LangGraph's stream into simple events (tool_call,
tool_result, nudge, final, error) so the web UI can show each step live.
"""

import json
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolErrorMiddleware,
    after_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from langgraph.types import Command

from papergraph.search import Paper

PROMPT_VERSION = "agent_system_v2"  # v2: adds papers_by_author
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
MAX_STEPS = 6  # model calls before tools are withheld and an answer is forced
MIN_SEARCHES = 2  # small models tend to stop after one search; nudge them once...
MIN_CITATIONS = 3  # ...but only if the answer is thin (fewer verified citations)
NUDGE = (
    "Before answering: you have made only one tool call. If the question has "
    "aspects your results do not cover yet, look them up with different, "
    "specific queries; if it is fully answered, give your final answer now."
)
STEP_LIMIT = (
    "Step limit reached. Write your final answer now, citing only papers from "
    "the tool results above."
)
REPEAT = (
    "You already made this exact call; its results are above. "
    "Use different arguments or write your final answer."
)


@dataclass
class Event:
    type: str  # thinking | tool_call | tool_result | nudge | final | error
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"type": self.type, **self.data})


def system_prompt(version: str = PROMPT_VERSION) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text()


_CITATION = re.compile(r"\[([A-Za-z0-9.\-/]+)\]")


def cited_ids(answer: str) -> list[str]:
    """Paper ids cited as [id] in the answer, in order, without repeats."""
    return list(dict.fromkeys(_CITATION.findall(answer)))


def call_key(name: str, arguments: dict[str, Any]) -> str:
    return name + json.dumps(arguments, sort_keys=True, default=str).lower()


def seen_papers(messages: Sequence[BaseMessage]) -> dict[str, Paper]:
    """Every paper the tools returned so far, by id (from the ToolMessage artifacts)."""
    papers: dict[str, Paper] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.artifact, list):
            papers.update((p.id, p) for p in m.artifact if isinstance(p, Paper))
    return papers


def verified(answer: str, papers: dict[str, Paper]) -> list[str]:
    return [i for i in cited_ids(answer) if i in papers]


def _text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


# --- guardrails (LangChain middleware) ------------------------------------------------


def _step_limit(max_steps: int) -> Any:
    @wrap_model_call
    def step_limit(
        request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        model_calls = sum(isinstance(m, AIMessage) for m in request.messages)
        if model_calls >= max_steps:
            # Withhold the tools: the model can only answer now.
            request = request.override(
                tools=[], messages=[*request.messages, HumanMessage(STEP_LIMIT)]
            )
        return handler(request)

    return step_limit


@wrap_tool_call
def no_repeats(
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]
) -> ToolMessage | Command[Any]:
    call = request.tool_call
    key = call_key(call["name"], call["args"])
    earlier = [
        tc
        for m in request.state["messages"]
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
        if tc["id"] != call["id"]
    ]
    if any(call_key(tc["name"], tc["args"]) == key for tc in earlier):
        return ToolMessage(REPEAT, tool_call_id=call["id"], name=call["name"], status="error")
    return handler(request)


def _nudge(min_searches: int, max_steps: int) -> Any:
    @after_model(can_jump_to=["model"])
    def nudge(state: AgentState[Any], runtime: Runtime[None]) -> dict[str, Any] | None:
        messages = state["messages"]
        last = messages[-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None  # not an answer: the loop continues normally
        n_calls = sum(isinstance(m, ToolMessage) for m in messages)
        already = any(isinstance(m, HumanMessage) and m.content == NUDGE for m in messages)
        model_calls = sum(isinstance(m, AIMessage) for m in messages)
        thin = len(verified(_text(last), seen_papers(messages))) < MIN_CITATIONS
        if n_calls < min_searches and thin and not already and model_calls < max_steps:
            return {"messages": [HumanMessage(NUDGE)], "jump_to": "model"}  # once
        return None

    return nudge


def _tool_error(exc: Exception, *args: Any, **kwargs: Any) -> str:
    return f"Error running the tool: {type(exc).__name__}: {exc}"


def build_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    max_steps: int = MAX_STEPS,
    min_searches: int = MIN_SEARCHES,
) -> Any:
    return create_agent(
        model,
        tools,
        system_prompt=system_prompt(),
        middleware=[
            ToolErrorMiddleware(on_error=_tool_error),
            _step_limit(max_steps),
            no_repeats,
            _nudge(min_searches, max_steps),
            # A hard backstop well above the normal flow (steps + nudge + forced answer).
            ModelCallLimitMiddleware(run_limit=max_steps + 3, exit_behavior="end"),
        ],
    )


# --- running it -------------------------------------------------------------------------


def _friendly(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ConnectionError) or "connect" in text.lower():
        return f"cannot reach Ollama ({text}); is `ollama serve` running?"
    return text


def run_agent(
    question: str,
    model: BaseChatModel,
    tools: list[BaseTool],
    max_steps: int = MAX_STEPS,
    min_searches: int = MIN_SEARCHES,
) -> Iterator[Event]:
    """Run the agent on `question`, yielding an Event for each step."""
    agent = build_agent(model, tools, max_steps, min_searches)
    start = time.perf_counter()
    history: list[BaseMessage] = []
    step = 0
    try:
        updates = agent.stream({"messages": [HumanMessage(question)]}, stream_mode="updates")
        for update in updates:
            for node_output in update.values():
                for m in (node_output or {}).get("messages", []):
                    history.append(m)
                    if isinstance(m, AIMessage):
                        step += 1
                        reasoning = m.additional_kwargs.get("reasoning_content")
                        if reasoning:
                            yield Event("thinking", {"step": step, "text": reasoning})
                        for tc in m.tool_calls:
                            yield Event(
                                "tool_call",
                                {"step": step, "name": tc["name"], "arguments": tc["args"]},
                            )
                    elif isinstance(m, ToolMessage):
                        returned = m.artifact if isinstance(m.artifact, list) else []
                        yield Event(
                            "tool_result",
                            {
                                "step": step,
                                "name": m.name,
                                "n_papers": len(returned),
                                "papers": [p.to_dict() for p in returned],
                                "error": _text(m) if m.status == "error" else None,
                            },
                        )
                    elif isinstance(m, HumanMessage) and m.content == NUDGE:
                        yield Event("nudge", {"step": step, "message": NUDGE})
    except Exception as e:  # the model server is down, a timeout, ...
        yield Event("error", {"message": _friendly(e)})
        return

    answers = [m for m in history if isinstance(m, AIMessage) and not m.tool_calls]
    if not answers:
        yield Event("error", {"message": "The agent stopped without an answer."})
        return
    papers = seen_papers(history)
    answer = _text(answers[-1])
    if len(answers) > 1:  # nudged: keep the first answer if the nudge made it worse
        first = _text(answers[0])
        if len(verified(first, papers)) > len(verified(answer, papers)):
            answer = first
    ids = cited_ids(answer)
    yield Event(
        "final",
        {
            "answer": answer,
            "papers": [papers[i].to_dict() for i in ids if i in papers],
            "unverified_ids": [i for i in ids if i not in papers],
            "steps": sum(isinstance(m, ToolMessage) for m in history),  # tool calls made
            "seconds": round(time.perf_counter() - start, 1),
            "prompt_version": PROMPT_VERSION,
        },
    )
