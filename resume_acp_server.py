"""ACP server: resume_tailor (CrewAI) on port 8003.

Produces resume-improvement suggestions tailored to the top-ranked job. It rewrites and
reframes ONLY what the candidate already provided — it never invents experience.
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
async def resume_tailor(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Suggest resume improvements tailored to the top-ranked job."""

    # --- ACP input boundary: client sends profile + top job in one message ---
    combined = input[0].parts[0].content

    agent = Agent(
        role="Resume Writer",
        goal="Tailor the candidate's resume to the target job without fabricating experience",
        backstory=(
            "You are an honest resume writer. You sharpen wording, surface relevant keywords, and "
            "reframe real projects — but you never invent jobs, titles, or achievements the "
            "candidate did not mention."
        ),
        llm=llm,
    )

    task = Task(
        description=(
            "Using the candidate profile and the target job below, write tailored resume "
            "suggestions. Base every bullet strictly on what the candidate actually provided. "
            "Where the job wants something the candidate lacks, say so in 'Warnings' instead of "
            "fabricating it.\n\n"
            f"{combined}"
        ),
        expected_output=(
            "Plain text using EXACTLY these section headings:\n"
            "Tailored Resume Summary: <2-3 sentence professional summary>\n"
            "Suggested Skills Section: <comma-separated, grounded in the profile>\n"
            "Improved Bullet Points: <3 to 6 bullets, each starting with a strong verb>\n"
            "Keywords To Include: <comma-separated keywords from the job posting>\n"
            "Warnings: <missing experience or any claim the profile does not support>"
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = await crew.kickoff_async()

    # --- ACP output boundary ---
    yield Message(parts=[MessagePart(content=str(result))])


if __name__ == "__main__":
    server.run(port=8003)
