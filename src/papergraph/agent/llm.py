"""The agent's language model: a local model served by Ollama (https://ollama.com),
used through LangChain's `ChatOllama`.

Any LangChain chat model with tool calling works in the agent, so another
backend (e.g. `ChatAnthropic`) is a one-line swap here.
"""

from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://localhost:11434"


def chat_model(
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    think: bool = False,
    num_ctx: int = 16384,
) -> ChatOllama:
    """The agent's chat model.

    - temperature 0: the same question gives the same answer.
    - num_ctx: Ollama's default context is small (a few thousand tokens) and it
      silently drops the oldest messages beyond it; an agent's history of tool
      results needs more.
    - reasoning (`think`): Qwen3 can reason before answering. Off by default:
      much faster, and the agent works well without it.
    """
    return ChatOllama(model=model, base_url=host, temperature=0, num_ctx=num_ctx, reasoning=think)
