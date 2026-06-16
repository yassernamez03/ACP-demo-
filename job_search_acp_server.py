"""ACP server: job_searcher (Smolagents) on port 8001.

Searches the web (DuckDuckGo via ddgs) for job openings that fit the structured candidate
profile produced by profile_analyzer, and returns a parseable job list.

Design note: this uses Smolagents' DuckDuckGoSearchTool + LiteLLMModel directly rather than
an autonomous CodeAgent loop. A CodeAgent kept writing brittle Python to *parse* the search
snippets — which on a local model crashed (undefined-variable InterpreterErrors) and emitted
10 identical rows, while burning ~3 minutes per step. The search itself was never the problem;
the result text is already grounded. So we call the tool directly for a couple of queries,
then make ONE model call to format those real results into the template. Same Smolagents
building blocks, but reliable and fast (one round-trip instead of a multi-step loop).
"""

import asyncio
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from smolagents import DuckDuckGoSearchTool, LiteLLMModel

warnings.filterwarnings("ignore")

server = Server()

# Default model. For weaker machines, switch to "ollama_chat/qwen2.5:7b".
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama_chat/qwen2.5:14b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LiteLLM (used here) DOES accept Ollama's num_ctx, unlike CrewAI's OpenAI-compatible path.
model = LiteLLMModel(model_id=OLLAMA_MODEL, api_base=OLLAMA_BASE_URL, num_ctx=8192)
search_tool = DuckDuckGoSearchTool()


def _derive_queries(profile: str) -> list[str]:
    """Build 1-2 short web-search queries from the structured profile.

    Reads the 'Target Role' / 'Location Preference' lines that profile_analyzer emits, with
    plain fallbacks so a format change never produces an empty query.
    """
    role = None
    location = None
    for line in profile.splitlines():
        low = line.lower()
        if low.startswith("target role:"):
            role = line.split(":", 1)[1].strip()
        elif low.startswith("location preference:"):
            location = line.split(":", 1)[1].strip()

    role = role or "backend developer"
    location = location or ""
    return [
        f"{role} junior internship {location}".strip(),
        f"{role} remote junior developer jobs".strip(),
    ]


def _search_and_format(profile: str) -> str:
    """Run the searches and the single formatting call. Blocking; run via asyncio.to_thread."""
    blocks = []
    for query in _derive_queries(profile):
        try:
            blocks.append(search_tool(query))
        except Exception as exc:  # one bad query (e.g. rate limit) shouldn't sink the rest
            blocks.append(f"(search failed for '{query}': {exc})")

    combined = "\n\n".join(b for b in blocks if b).strip()
    if not combined:
        return (
            "No job results were returned by the web search (the search backend may be rate-"
            "limited). Try re-running in a minute."
        )

    prompt = (
        "You are a job-search assistant. Below are RAW web search results gathered for a "
        "candidate. Each result is a Markdown link in the form [Job Title](URL) followed by a "
        "description.\n\n"
        f"CANDIDATE PROFILE:\n{profile}\n\n"
        f"RAW SEARCH RESULTS:\n{combined}\n\n"
        "From the results above, extract 5 to 10 DISTINCT real job openings that fit the "
        "candidate. Every entry MUST be a different posting — never repeat the same company+title "
        "or the same URL. Use ONLY information present in the results; if a field is unknown write "
        "N/A. For the URL field output the raw link only (the text inside the parentheses of the "
        "[..](URL) markdown) with no surrounding brackets; write N/A only if there is genuinely no "
        "link. Do NOT invent companies or URLs.\n"
        "Return ONLY a numbered list, each job in EXACTLY this format:\n"
        "1. Title: <title>\n"
        "   Company: <company>\n"
        "   Location/Remote: <city/country or Remote/Hybrid>\n"
        "   URL: <url or N/A>\n"
        "   Why relevant: <one sentence tying it to the candidate>\n"
        "   Required skills: <comma-separated skills from the posting>\n"
    )

    response = model.generate([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
    return response.content


@server.agent()
async def job_searcher(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Search the web for job openings matching the candidate profile."""

    # --- ACP input boundary: the incoming message is the structured candidate profile ---
    profile = input[0].parts[0].content

    # The search + formatting are blocking (network + LLM); run off the event loop so the
    # server stays responsive (e.g. to /agents health checks) while it works.
    result = await asyncio.to_thread(_search_and_format, profile)

    # --- ACP output boundary: return the job list as a structured ACP message ---
    yield Message(parts=[MessagePart(content=result)])


if __name__ == "__main__":
    server.run(port=8001)
