"""ACP server: interview_coach (Smolagents) on port 8005.

Generates an interview-prep pack for the top-ranked job. This is a pure text-generation
task, so the CodeAgent runs with NO tools and is told to return the finished pack
directly via final_answer (otherwise a no-tools CodeAgent wastes steps writing scaffolding).

NOTE (escape hatch): if this agent is flaky against your local Ollama, this is the one
file to swap to a CrewAI Agent/Task — no other file needs to change.
"""

import os
import warnings

from dotenv import load_dotenv

load_dotenv()

from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from smolagents import CodeAgent, LiteLLMModel

warnings.filterwarnings("ignore")

server = Server()

# Default model. For weaker machines, switch to "ollama_chat/qwen2.5:7b".
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama_chat/qwen2.5:14b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

model = LiteLLMModel(model_id=OLLAMA_MODEL, api_base=OLLAMA_BASE_URL, num_ctx=8192)


@server.agent()
async def interview_coach(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Create interview-prep material for the top-ranked job."""

    # --- ACP input boundary: client sends profile + top job + skills gap ---
    combined = input[0].parts[0].content

    # No tools: this is generation, not research. Keep max_steps low so the agent emits
    # its answer promptly instead of looping.
    agent = CodeAgent(tools=[], model=model, max_steps=3)

    prompt = (
        "You are an interview coach preparing a candidate for a specific job. Using the candidate "
        "profile, target job, and skills gap below, build an interview-prep pack.\n\n"
        f"{combined}\n\n"
        "Do NOT write or run any code and do NOT call tools. Compose the full pack as plain text "
        "and return it directly via final_answer(). Use EXACTLY these sections:\n"
        "Likely Interview Questions: <8 questions, numbered>\n"
        "Answer Strategy: <for each of the 8 questions, one short strategy line>\n"
        "Technical Topics To Review: <exactly 3>\n"
        "Questions To Ask The Employer: <exactly 3>"
    )

    # Smolagents .run is synchronous.
    response = agent.run(prompt)

    # --- ACP output boundary ---
    yield Message(parts=[MessagePart(content=str(response))])


if __name__ == "__main__":
    server.run(port=8005)
