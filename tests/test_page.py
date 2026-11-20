"""Guards for the web page (src/papergraph/api/static/index.html).

There are no JavaScript tests, so these check the lines the page cannot work
without. Background: an edit once commented out the line that opens the agent
stream; the button then showed "Planning searches" forever with no error.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).parents[1] / "src/papergraph/api/static/index.html"
SCRIPT = re.search(r"<script>(.*)</script>", PAGE.read_text(), re.S)


def live_code(pattern: str) -> bool:
    """Whether `pattern` appears on a line of the page's script that is not a comment."""
    assert SCRIPT is not None
    lines = (line.strip() for line in SCRIPT.group(1).splitlines())
    return any(re.search(pattern, line) for line in lines if not line.startswith("//"))


@pytest.mark.parametrize(
    "pattern",
    [
        r'source = new EventSource\("/api/agent\?',  # the agent stream
        r"source\.onmessage = ",  # handling its events
        r'fetch\("/api/search\?"',  # search results
        r'fetch\("/api/meta"\)',  # filters and corpus info
    ],
)
def test_essential_code_is_not_commented_out(pattern: str) -> None:
    assert live_code(pattern), f"missing or commented out in index.html: {pattern}"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_script_is_valid_javascript(tmp_path: Path) -> None:
    assert SCRIPT is not None
    script = tmp_path / "page.js"
    script.write_text(SCRIPT.group(1))
    result = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
