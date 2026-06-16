"""ACP server: job_searcher (Smolagents) on port 8001.

Searches the web (DuckDuckGo via ddgs) for job openings that fit the structured
candidate profile produced by profile_analyzer, and returns a parseable job list.
"""

import os
import warnings

from dotenv import load_dotenv

load_dotenv()

from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

warnings.filterwarnings("ignore")

server = Server()

# Default model. For weaker machines, switch to "ollama_chat/qwen2.5:7b".
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama_chat/qwen2.5:14b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

model = LiteLLMModel(model_id=OLLAMA_MODEL, api_base=OLLAMA_BASE_URL, num_ctx=8192)


@server.agent()
async def job_searcher(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Search the web for job openings matching the candidate profile."""

    # --- ACP input boundary: the incoming message is the structured candidate profile ---
    profile = input[0].parts[0].content

    # CodeAgent can call DuckDuckGoSearchTool to gather real listings, then summarize them.
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

    prompt = (
        "You are a job-search assistant. Using the DuckDuckGo search tool, find 5 to 10 real, "
        "relevant job openings for the candidate described below. Prefer recent postings and "
        "respect their location/remote preference.\n\n"
        f"CANDIDATE PROFILE:\n{profile}\n\n"
        "If a search returns few results, broaden the query (synonyms for the role, nearby "
        "regions, 'junior'/'internship' variants) rather than returning nothing.\n\n"
        "Return ONLY a numbered list of jobs. For EACH job use exactly this template:\n"
        "1. Title: <job title>\n"
        "   Company: <company>\n"
        "   Location/Remote: <city/country or Remote/Hybrid>\n"
        "   URL: <link, or 'N/A' if unknown>\n"
        "   Why relevant: <one sentence tying it to the candidate>\n"
        "   Required skills: <comma-separated skills mentioned in the posting>\n"
        "Do not fabricate companies or URLs; use 'N/A' when unsure. Provide your full job list "
        "as the final answer."
    )

    # Smolagents .run is synchronous; it performs the search + reasoning loop internally.
    response = agent.run(prompt)

    # --- ACP output boundary: return the job list as a structured ACP message ---
    yield Message(parts=[MessagePart(content=str(response))])


if __name__ == "__main__":
    server.run(port=8001)
