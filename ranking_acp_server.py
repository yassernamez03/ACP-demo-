"""ACP server: job_ranker (CrewAI) on port 8002.

Compares the candidate profile against the job-search results and ranks the jobs from
best to weakest match. Ends its output with a machine-readable TOP_JOB block so the
client can reliably extract the single best job for the downstream agents.
"""

import os

os.environ["CREWAI_TRACING_ENABLED"] = "false"

from collections.abc import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from crewai import LLM, Agent, Crew, Task

server = Server()

# Default model. For weaker machines, switch to "ollama_chat/qwen2.5:7b".
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama_chat/qwen2.5:14b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# max_tokens caps the response length. NOTE: this CrewAI version reaches Ollama through its
# OpenAI-compatible endpoint, which does NOT accept Ollama's num_ctx option (passing it raises
# "unexpected keyword argument 'num_ctx'"). To enlarge the input context window, set Ollama's
# OLLAMA_CONTEXT_LENGTH env var instead of passing num_ctx here.
llm = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, max_tokens=8192)


@server.agent()
async def job_ranker(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Rank job-search results against the candidate profile."""

    # --- ACP input boundary: the client sends profile + job results in one message ---
    combined = input[0].parts[0].content

    agent = Agent(
        role="Technical Recruiter",
        goal="Rank job openings by how well they fit the candidate, honestly and specifically",
        backstory=(
            "You are a pragmatic recruiter who matches candidates to roles. You reward genuine "
            "fit, flag real concerns, and never inflate a match score to be encouraging."
        ),
        llm=llm,
    )

    task = Task(
        description=(
            "Below are a candidate profile and a list of job openings. Rank the jobs from best "
            "match to weakest match for THIS candidate.\n\n"
            f"{combined}\n\n"
            "Score honestly against the candidate's actual skills and constraints."
        ),
        expected_output=(
            "Plain text with two parts.\n\n"
            "PART 1 — a ranked list (best first). For each job include:\n"
            "  - Rank and Title @ Company\n"
            "  - Match score: <0-100>\n"
            "  - Reasons for match: <short>\n"
            "  - Potential concerns: <short>\n"
            "  - Missing skills: <comma-separated>\n"
            "Then a 'Recommended Top 3' summary line.\n\n"
            "PART 2 — end the output with EXACTLY this machine-readable block describing the #1 "
            "ranked job (keep the marker lines exactly as shown, on their own lines):\n"
            "===TOP_JOB_START===\n"
            "Title: <title>\n"
            "Company: <company>\n"
            "Location/Remote: <location or remote>\n"
            "URL: <url or N/A>\n"
            "Match score: <0-100>\n"
            "Why it's #1: <one or two sentences>\n"
            "Missing skills: <comma-separated>\n"
            "===TOP_JOB_END==="
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = await crew.kickoff_async()

    # --- ACP output boundary ---
    yield Message(parts=[MessagePart(content=str(result))])


if __name__ == "__main__":
    server.run(port=8002)
