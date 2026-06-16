"""ACP server: profile_analyzer (CrewAI) on port 8000.

Analyzes a raw job-search request and extracts a structured candidate profile that
every downstream agent (search, ranking, resume, cover letter, interview) builds on.
"""

import os

# Hard-set BEFORE importing crewai so we never hit the interactive trace prompt that
# would otherwise block the server from starting.
os.environ["CREWAI_TRACING_ENABLED"] = "false"

from collections.abc import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()  # override=False by default -> won't flip our "false" back to "true"

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
async def profile_analyzer(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Analyze a candidate's job-search request into a structured profile."""

    # --- ACP input boundary: read the user's request text from the incoming message ---
    request = input[0].parts[0].content

    agent = Agent(
        role="Career Profile Analyst",
        goal="Turn a candidate's free-text job-search request into a clear, structured profile",
        backstory=(
            "You are a precise career coach. You read a candidate's request and extract exactly "
            "what they want and what they bring, without inventing facts. If something is unclear "
            "or absent, you call it out rather than guessing."
        ),
        llm=llm,
    )

    task = Task(
        description=(
            "Analyze the following candidate job-search request and produce a structured profile.\n\n"
            f"CANDIDATE REQUEST:\n{request}\n\n"
            "Do NOT invent experience or skills. If information is missing, list it under "
            "'Missing Information' instead of guessing."
        ),
        expected_output=(
            "Plain text using EXACTLY these section headings, each on its own line:\n"
            "Target Role: <one line>\n"
            "Seniority: <e.g. junior / mid / senior, plus internships if relevant>\n"
            "Location Preference: <remote/hybrid/onsite and regions>\n"
            "Core Skills: <comma-separated list grounded in the request>\n"
            "Strengths: <2-4 short bullets>\n"
            "Constraints: <2-4 short bullets, e.g. no professional experience yet>\n"
            "Missing Information: <what the candidate did not provide that a recruiter would want>"
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = await crew.kickoff_async()

    # --- ACP output boundary: return the analysis as a structured ACP message ---
    yield Message(parts=[MessagePart(content=str(result))])


if __name__ == "__main__":
    server.run(port=8000)
