from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from chainlit.context import ChainlitContext, context_var
from chainlit.message import AskUserMessage
from chainlit.step import StepDict
from chainlit.sync import run_sync
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


AskUserCallable: TypeAlias = Callable[[str], str | None]
ASK_USER_TOOL_NAME = "Ask user a follow-up question"


class ChainlitAskUserTimeoutError(RuntimeError):
    """Raised when the user does not answer before the configured timeout."""


class AskUserQuestionInput(BaseModel):
    """Input schema for asking the user a single follow-up question."""

    question: str = Field(
        ...,
        min_length=1,
        description=(
            "A single, clear follow-up question for the user. Ask one thing at a time."
        ),
    )


def _extract_answer_text(response: StepDict | None) -> str | None:
    if response is None:
        return None

    output = response.get("output")
    if output is None:
        return None
    if isinstance(output, str):
        normalized_output = output.strip()
        return normalized_output or None
    return str(output)


def _resolve_chainlit_context(
    chainlit_context: ChainlitContext | None,
) -> ChainlitContext:
    if chainlit_context is not None:
        return chainlit_context

    try:
        return context_var.get()
    except LookupError as exc:
        raise RuntimeError(
            "ChainlitAskUserTool must be created inside an active Chainlit request "
            "or passed an explicit `chainlit_context`."
        ) from exc


def _build_default_ask_user(
    *,
    chainlit_context: ChainlitContext,
    timeout_seconds: int,
    author: str,
    raise_on_timeout: bool,
) -> AskUserCallable:
    def ask_user(question: str) -> str | None:
        token = context_var.set(chainlit_context)
        try:
            response = run_sync(
                AskUserMessage(
                    content=question,
                    author=author,
                    timeout=timeout_seconds,
                    raise_on_timeout=raise_on_timeout,
                ).send()
            )
        except TimeoutError as exc:
            raise ChainlitAskUserTimeoutError(
                f"The user did not answer within {timeout_seconds} seconds."
            ) from exc
        finally:
            context_var.reset(token)

        return _extract_answer_text(response)

    return ask_user


class ChainlitAskUserTool(BaseTool):
    """CrewAI tool that asks the active Chainlit user a follow-up question."""

    name: str = ASK_USER_TOOL_NAME
    description: str = (
        "Ask the human user a single follow-up question when more context is needed. "
        "Use this to clarify goals, deadlines, preferences, or missing details before "
        "continuing with the task."
    )
    args_schema: type[BaseModel] = AskUserQuestionInput

    timeout_seconds: int = Field(default=120, ge=1, exclude=True)
    author: str = Field(default="Crew Assistant", exclude=True)
    fail_on_timeout: bool = Field(default=False, exclude=True)
    empty_response_message: str = Field(
        default="No user response was received before the timeout.",
        exclude=True,
    )

    _ask_user: AskUserCallable = PrivateAttr()

    def __init__(
        self,
        *,
        ask_user: AskUserCallable | None = None,
        chainlit_context: ChainlitContext | None = None,
        timeout_seconds: int = 120,
        author: str = "Crew Assistant",
        fail_on_timeout: bool = False,
        empty_response_message: str = "No user response was received before the timeout.",
    ) -> None:
        super().__init__(
            timeout_seconds=timeout_seconds,
            author=author,
            fail_on_timeout=fail_on_timeout,
            empty_response_message=empty_response_message,
        )
        self._ask_user = ask_user or _build_default_ask_user(
            chainlit_context=_resolve_chainlit_context(chainlit_context),
            timeout_seconds=timeout_seconds,
            author=author,
            raise_on_timeout=fail_on_timeout,
        )

    def _run(self, question: str) -> str:
        answer = self._ask_user(question)
        if answer is not None and answer.strip():
            return answer.strip()
        if self.fail_on_timeout:
            raise ChainlitAskUserTimeoutError(self.empty_response_message)
        return self.empty_response_message
