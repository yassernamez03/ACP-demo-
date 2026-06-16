"""ACP server: interview_coach (CrewAI) on port 8005.

Generates an interview-prep pack for the top-ranked job: likely questions with answer
strategies, technical topics to review, and questions for the candidate to ask the employer.

This agent uses CrewAI. (It originally used a no-tools Smolagents CodeAgent, but for pure
prose generation that agent leaked literal `final_answer(...)` wrapper text; CrewAI produces
clean prose reliably, like the other text-generation agents. job_search_acp_server.py remains
the Smolagents example, since it genuinely needs the DuckDuckGo search tool.)
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
async def interview_coach(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Create interview-prep material for the top-ranked job."""

    # --- ACP input boundary: client sends profile + top job + skills gap ---
    combined = input[0].parts[0].content

    agent = Agent(
        role="Interview Coach",
        goal="Prepare the candidate for a specific job interview with realistic, tailored material",
        backstory=(
            "You are a supportive but honest interview coach. You anticipate the questions this "
            "candidate will actually face for this role, and you give concrete strategies grounded "
            "in their real background rather than generic advice."
        ),
        llm=llm,
    )

    task = Task(
        description=(
            "Using the candidate profile, target job, and skills gap below, build a complete "
            "interview-prep pack tailored to THIS candidate and THIS role.\n\n"
            f"{combined}\n\n"
            "Make the questions and strategies specific to the candidate's actual skills and the "
            "job's requirements. Do not invent experience the candidate did not mention."
        ),
        expected_output=(
            "Plain text using EXACTLY these section headings, each on its own line:\n"
            "Likely Interview Questions: <8 questions, numbered 1-8>\n"
            "Answer Strategy: <for each of the 8 questions, one short strategy line, numbered 1-8>\n"
            "Technical Topics To Review: <exactly 3, as a short list>\n"
            "Questions To Ask The Employer: <exactly 3, as a short list>"
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = await crew.kickoff_async()

    # --- ACP output boundary ---
    yield Message(parts=[MessagePart(content=str(result))])


if __name__ == "__main__":
    server.run(port=8005)
