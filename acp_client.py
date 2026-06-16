import asyncio

from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart


def text_message(content: str) -> list[Message]:
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


async def run_workflow() -> None:
    topic = "Impact of climate change on agriculture in 2025."

    async with Client(
        base_url="http://localhost:8000",
        headers={"Content-Type": "application/json"},
    ) as drafter:
        response1 = await drafter.run_sync(
            agent="research_drafter",
            input=text_message(topic),
        )

    draft = response1.output[0].parts[0].content
    print(f"\nDraft Summary:\n{draft}")

    async with Client(
        base_url="http://localhost:8001",
        headers={"Content-Type": "application/json"},
    ) as verifier:
        response2 = await verifier.run_sync(
            agent="research_verifier",
            input=text_message(
                "Enhance the following research summary using the latest information "
                "available online, providing a more accurate and updated version:\n\n"
                f"{draft}"
            ),
        )

    final_summary = response2.output[0].parts[0].content
    print(f"\nVerified & Enriched Summary:\n{final_summary}")


if __name__ == "__main__":
    asyncio.run(run_workflow())
