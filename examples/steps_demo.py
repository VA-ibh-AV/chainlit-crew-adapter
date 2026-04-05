import os
import re
import sys
from pathlib import Path

import chainlit as cl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chainlit_crew_adapter import ChainlitCrewAdapter
from examples.crews.steps_demo import build_steps_demo_crew


def extract_post_id(message_content: str) -> int:
    """Extract a JSONPlaceholder post id from the message or fall back to 1."""
    match = re.search(r"\b([1-9][0-9]?)\b", message_content)
    if not match:
        return 1

    candidate = int(match.group(1))
    if candidate > 100:
        return 1
    return candidate


def format_crew_result(result: object) -> str:
    """Return a user-facing string from the crew result."""
    raw_output = getattr(result, "raw", None)
    if isinstance(raw_output, str):
        return raw_output
    return str(result)


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content=(
            "Ask for a post analysis like `Analyze post 5` or `Summarize post 12`. "
            "This demo shows task, agent, and tool steps in the Chainlit trace."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        await cl.Message(
            content=(
                "Set `OPENAI_API_KEY` before running this example so the crew can call "
                "the model."
            )
        ).send()
        return

    post_id = extract_post_id(message.content)
    crew = build_steps_demo_crew()
    adapter = ChainlitCrewAdapter(cl=cl, crew=crew)
    result = await adapter.kickoff(
        inputs={
            "user_request": message.content,
            "post_id": post_id,
        }
    )
    await cl.Message(content=format_crew_result(result)).send()
