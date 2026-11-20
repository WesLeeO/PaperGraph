"""Fake LangChain chat models for tests: scripted replies, no Ollama needed."""

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field


class FakeModel(FakeMessagesListChatModel):
    """Returns scripted replies in order and records the messages of every call."""

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeModel":
        return self  # the replies are scripted: the tools on offer don't matter

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


class DownModel(BaseChatModel):
    """A model whose server is unreachable."""

    @property
    def _llm_type(self) -> str:
        return "down"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DownModel":
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        raise ConnectionError("Failed to connect to Ollama")
