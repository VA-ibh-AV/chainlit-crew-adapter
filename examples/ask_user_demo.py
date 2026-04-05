import os
import sys
from pathlib import Path

import chainlit as cl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chainlit_crew_adapter import ChainlitAskUserTool, ChainlitCrewAdapter
from examples.crews.ask_user_demo import build_ask_user_demo_crew


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
            "Share a request like `Help me plan a product launch` or `I need a landing "
            "page brief`. The crew will ask a couple of follow-up questions before it "
            "writes the final answer."
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

    ask_user_tool = ChainlitAskUserTool(timeout_seconds=180, author="Crew Assistant")
    crew = build_ask_user_demo_crew(ask_user_tool=ask_user_tool)
    adapter = ChainlitCrewAdapter(cl=cl, crew=crew, show_agent_steps=False)
    result = await adapter.kickoff(inputs={"user_request": message.content})
    await cl.Message(content=format_crew_result(result)).send()
