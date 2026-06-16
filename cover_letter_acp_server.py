"""ACP server: cover_letter_writer (CrewAI) on port 8004.

Drafts a concise, tailored cover letter for the top-ranked job using the candidate
profile and resume-tailoring suggestions. Uses [Placeholders] instead of inventing facts.
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

# num_ctx = input context window (Ollama defaults small and would silently truncate the
# multi-section inputs these agents receive); max_tokens = output length.
llm = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, max_tokens=8192, num_ctx=8192)


@server.agent()
async def cover_letter_writer(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Draft a tailored cover letter for the top-ranked job."""

    # --- ACP input boundary: client sends profile + top job + resume suggestions ---
    combined = input[0].parts[0].content

    agent = Agent(
        role="Cover Letter Writer",
        goal="Write a concise, sincere, tailored cover letter that does not invent facts",
        backstory=(
            "You write cover letters that sound human and specific. You connect the candidate's "
            "real strengths to the role. When a concrete detail is missing, you insert a clearly "
            "marked placeholder rather than making something up."
        ),
        llm=llm,
    )

    task = Task(
        description=(
            "Write a professional cover letter for the candidate applying to the target job, "
            "using the profile and resume suggestions below.\n\n"
            f"{combined}\n\n"
            "Keep it to roughly 250-350 words, realistic and grounded. Wherever a fact is not "
            "provided, use an explicit placeholder such as [Company Name], [Hiring Manager], or "
            "[Specific Achievement]. Do NOT invent achievements or experience."
        ),
        expected_output=(
            "A complete cover letter in plain text: greeting, 3-4 short paragraphs, and a sign-off. "
            "Any missing facts appear as bracketed [Placeholders]."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = await crew.kickoff_async()

    # --- ACP output boundary ---
    yield Message(parts=[MessagePart(content=str(result))])


if __name__ == "__main__":
    server.run(port=8004)
