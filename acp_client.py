"""ACP client / orchestrator for the Job Search Assistant demo.

This is where the ACP communication happens: the client opens an ACP `Client` against
each agent's local server and calls `run_sync`. Each agent runs on its own port and may
be built with a different framework (CrewAI or Smolagents) — ACP is what lets them
collaborate. The client wires their outputs together into one consolidated report.

Run order (each step talks to a different ACP server):
  [1/7] profile_analyzer    -> structured candidate profile
  [2/7] job_searcher        -> list of job openings
  [3/7] job_ranker          -> ranked jobs + a TOP_JOB block
  [4/7] resume_tailor       -> resume suggestions for the top job
  [5/7] cover_letter_writer -> cover letter for the top job
  [6/7] interview_coach     -> interview prep for the top job
  [7/7] print consolidated report
"""

import asyncio
import sys

# Agent output (and this report) contains em-dashes, curly quotes, and bullets that the
# default Windows console encoding (cp1252) cannot encode -> printing would raise
# UnicodeEncodeError at the very end, after all six slow agent calls. Force UTF-8 so the
# consolidated report always prints (and survives being redirected to a log file).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import httpx
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart
from acp_sdk.models.errors import ACPError

# ---------------------------------------------------------------------------
# Default candidate profile. EDIT THIS (or replace it) to try your own search.
# ---------------------------------------------------------------------------
USER_PROFILE = """\
I am a junior Python developer looking for remote or hybrid backend developer roles in Europe or Morocco.
I have experience with Python, FastAPI, SQL, Git, Docker basics, and building small AI agent demos.
I am interested in backend engineering, AI engineering, and automation roles.
I do not have professional work experience yet, but I have academic projects and personal projects.
I want jobs that are friendly to junior developers or internships.
"""

# Agent name -> (base_url, port, server filename). The agent name must match the
# function name registered on the corresponding server.
AGENTS = {
    "profile_analyzer": ("http://localhost:8000", 8000, "profile_acp_server.py"),
    "job_searcher": ("http://localhost:8001", 8001, "job_search_acp_server.py"),
    "job_ranker": ("http://localhost:8002", 8002, "ranking_acp_server.py"),
    "resume_tailor": ("http://localhost:8003", 8003, "resume_acp_server.py"),
    "cover_letter_writer": ("http://localhost:8004", 8004, "cover_letter_acp_server.py"),
    "interview_coach": ("http://localhost:8005", 8005, "interview_acp_server.py"),
}

# Fail fast if a server isn't listening (connect), but allow long LLM generations (read).
# `run_sync` has no per-call timeout, so we configure it on the httpx client instead.
TIMEOUT = httpx.Timeout(timeout=None, connect=5.0)

TOP_JOB_START = "===TOP_JOB_START==="
TOP_JOB_END = "===TOP_JOB_END==="


def text_message(content: str) -> list[Message]:
    """Wrap plain text as a structured ACP message (never pass raw strings to ACP)."""
    return [
        Message(
            role="user",
            parts=[
                MessagePart(
                    content=content,
                    content_type="text/plain",
                )
            ],
        )
    ]


def get_text_output(response, agent_name: str) -> str:
    """Safely pull the text out of an ACP response, with clear errors when it's empty."""
    if not getattr(response, "output", None):
        raise RuntimeError(
            f"{agent_name} returned no output. Check that server terminal for the real error."
        )

    if not response.output[0].parts:
        raise RuntimeError(f"{agent_name} returned an output message with no parts.")

    return response.output[0].parts[0].content


async def call_agent(agent_name: str, content: str) -> str:
    """Send `content` to one ACP agent and return its text output.

    Raises SystemExit with a friendly message if the server is unreachable or errors,
    so the demo never dies with a raw traceback or IndexError.
    """
    base_url, port, server_file = AGENTS[agent_name]

    # --- ACP communication: one Client per agent server ---
    try:
        async with Client(
            base_url=base_url,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"},
        ) as client:
            response = await client.run_sync(agent=agent_name, input=text_message(content))
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise SystemExit(
            f"\n[X] Could not reach '{agent_name}' on :{port}.\n"
            f"    Is that server running? Start it in its own terminal with:\n"
            f"        uv run {server_file}\n"
        )
    except ACPError as err:
        raise SystemExit(
            f"\n[X] '{agent_name}' (:{port}) returned an ACP error: {err}\n"
            f"    Check the '{server_file}' terminal for the full traceback.\n"
        )

    return get_text_output(response, agent_name)


def extract_top_job(ranking_text: str) -> str:
    """Pull the delimited TOP_JOB block out of the ranking output.

    Falls back to the full ranking text if the markers are missing, so a malformed
    ranking never breaks the rest of the workflow.
    """
    start = ranking_text.find(TOP_JOB_START)
    end = ranking_text.find(TOP_JOB_END)
    if start != -1 and end != -1 and end > start:
        return ranking_text[start + len(TOP_JOB_START) : end].strip()

    print("    (note: TOP_JOB markers not found — passing the full ranking downstream)")
    return ranking_text.strip()


def section(title: str, body: str) -> str:
    bar = "=" * 78
    return f"\n{bar}\n{title}\n{bar}\n{body.strip()}\n"


async def run_workflow() -> None:
    print("ACP Job Search Assistant - orchestrating 6 agents across 6 local servers.\n")

    # [1/7] Analyze the candidate profile (CrewAI @ 8000).
    print("[1/7] Analyzing candidate profile...")
    profile_analysis = await call_agent("profile_analyzer", USER_PROFILE)

    # [2/7] Search for jobs based on the analyzed profile (Smolagents @ 8001).
    print("[2/7] Searching for jobs...")
    job_results = await call_agent("job_searcher", profile_analysis)

    # [3/7] Rank the jobs against the profile (CrewAI @ 8002).
    print("[3/7] Ranking jobs against the profile...")
    ranking = await call_agent(
        "job_ranker",
        f"CANDIDATE PROFILE:\n{profile_analysis}\n\nJOB SEARCH RESULTS:\n{job_results}",
    )

    # [4/7] Pick the single best job for the tailoring agents.
    top_job = extract_top_job(ranking)

    # [5/7] Tailor the resume to the top job (CrewAI @ 8003).
    print("[4/7] Tailoring resume for the top job...")
    resume_suggestions = await call_agent(
        "resume_tailor",
        f"CANDIDATE PROFILE:\n{profile_analysis}\n\nTOP RANKED JOB:\n{top_job}",
    )

    # [6/7] Draft a cover letter for the top job (CrewAI @ 8004).
    print("[5/7] Drafting cover letter...")
    cover_letter = await call_agent(
        "cover_letter_writer",
        f"CANDIDATE PROFILE:\n{profile_analysis}\n\nTOP RANKED JOB:\n{top_job}\n\n"
        f"RESUME TAILORING SUGGESTIONS:\n{resume_suggestions}",
    )

    # [7/7] Build interview prep for the top job (Smolagents @ 8005).
    print("[6/7] Preparing interview questions...")
    interview_prep = await call_agent(
        "interview_coach",
        f"CANDIDATE PROFILE:\n{profile_analysis}\n\nTOP RANKED JOB:\n{top_job}\n\n"
        f"RANKING & SKILLS GAP:\n{ranking}",
    )

    # Consolidated report.
    print("[7/7] Building final report...\n")
    report = (
        section("CANDIDATE PROFILE SUMMARY", profile_analysis)
        + section("JOB SEARCH RESULTS", job_results)
        + section("RANKED JOBS", ranking)
        + section("TOP RECOMMENDED JOB", top_job)
        + section("RESUME TAILORING SUGGESTIONS", resume_suggestions)
        + section("COVER LETTER DRAFT", cover_letter)
        + section("INTERVIEW PREP PACK", interview_prep)
    )
    print(report)


if __name__ == "__main__":
    asyncio.run(run_workflow())
