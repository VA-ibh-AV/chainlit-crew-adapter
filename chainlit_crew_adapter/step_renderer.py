from collections.abc import Mapping
from typing import Protocol

from chainlit.context import local_steps
from crewai.utilities.string_utils import sanitize_tool_name

from .events import (
    AdapterEvent,
    AgentCompletedUpdate,
    AgentFailedUpdate,
    AgentStartedUpdate,
    CrewCompletedUpdate,
    CrewFailedUpdate,
    CrewStartedUpdate,
    TaskCompletedUpdate,
    TaskFailedUpdate,
    TaskStartedUpdate,
    ToolArguments,
    ToolFailedUpdate,
    ToolFinishedUpdate,
    ToolStartedUpdate,
)
from .human_input import ASK_USER_TOOL_NAME


class ChainlitStep(Protocol):
    id: str
    input: object
    output: object
    is_error: bool
    metadata: dict[str, object]

    async def send(self) -> object: ...

    async def update(self) -> bool: ...


class ChainlitStepFactory(Protocol):
    def __call__(
        self,
        name: str = "",
        type: str = "undefined",
        id: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, object] | None = None,
        tags: list[str] | None = None,
        language: str | None = None,
        default_open: bool | None = False,
        auto_collapse: bool | None = False,
        show_input: bool | str = "json",
        thread_id: str | None = None,
    ) -> ChainlitStep: ...


class SupportsChainlitSteps(Protocol):
    Step: ChainlitStepFactory


def _current_parent_step_id() -> str | None:
    steps = local_steps.get() or []
    if not steps:
        return None
    return steps[-1].id


def _clean_label(label: str | None, fallback: str) -> str:
    text = (label or "").strip()
    if not text:
        return fallback
    single_line = " ".join(text.split())
    if len(single_line) <= 80:
        return single_line
    return f"{single_line[:77]}..."


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _is_ask_user_tool(tool_name: str) -> bool:
    return sanitize_tool_name(tool_name) == sanitize_tool_name(ASK_USER_TOOL_NAME)


def _extract_tool_question(tool_args: ToolArguments) -> str:
    if isinstance(tool_args, dict):
        question = tool_args.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    return _stringify(tool_args).strip()


class ChainlitStepRenderer:
    """Render adapter events into Chainlit steps."""

    def __init__(
        self,
        cl: SupportsChainlitSteps,
        crew_name: str | None,
        inputs: Mapping[str, object] | None,
        show_agent_steps: bool = True,
    ) -> None:
        self._cl = cl
        self._crew_name = crew_name or "Crew"
        self._inputs = dict(inputs or {})
        self._show_agent_steps = show_agent_steps
        self._root_parent_id = _current_parent_step_id()
        self._crew_step: ChainlitStep | None = None
        self._task_steps: dict[str, ChainlitStep] = {}
        self._agent_steps: dict[tuple[str, str], ChainlitStep] = {}
        self._tool_steps: dict[tuple[str | None, str | None, str, int], ChainlitStep] = (
            {}
        )
        self._clarification_steps: dict[str, ChainlitStep] = {}
        self._clarification_questions: dict[str, list[str]] = {}

    async def initialize(self) -> None:
        crew_step = self._cl.Step(
            name=self._crew_name,
            type="run",
            parent_id=self._root_parent_id,
            show_input="json",
            default_open=True,
            metadata={"component": "crew", "status": "pending"},
        )
        crew_step.input = self._inputs
        await crew_step.send()
        self._crew_step = crew_step

    async def handle_event(self, event: AdapterEvent) -> None:
        if isinstance(event, CrewStartedUpdate):
            await self._handle_crew_started(event)
        elif isinstance(event, CrewCompletedUpdate):
            await self._handle_crew_completed(event)
        elif isinstance(event, CrewFailedUpdate):
            await self._handle_crew_failed(event)
        elif isinstance(event, TaskStartedUpdate):
            await self._handle_task_started(event)
        elif isinstance(event, TaskCompletedUpdate):
            await self._handle_task_completed(event)
        elif isinstance(event, TaskFailedUpdate):
            await self._handle_task_failed(event)
        elif isinstance(event, AgentStartedUpdate):
            await self._handle_agent_started(event)
        elif isinstance(event, AgentCompletedUpdate):
            await self._handle_agent_completed(event)
        elif isinstance(event, AgentFailedUpdate):
            await self._handle_agent_failed(event)
        elif isinstance(event, ToolStartedUpdate):
            await self._handle_tool_started(event)
        elif isinstance(event, ToolFinishedUpdate):
            await self._handle_tool_finished(event)
        elif isinstance(event, ToolFailedUpdate):
            await self._handle_tool_failed(event)

    async def _handle_crew_started(self, event: CrewStartedUpdate) -> None:
        crew_step = self._require_crew_step()
        crew_step.metadata["status"] = "running"
        crew_step.input = event.inputs
        await crew_step.update()

    async def _handle_crew_completed(self, event: CrewCompletedUpdate) -> None:
        crew_step = self._require_crew_step()
        crew_step.metadata["status"] = "completed"
        crew_step.output = event.output
        await crew_step.update()

    async def _handle_crew_failed(self, event: CrewFailedUpdate) -> None:
        crew_step = self._require_crew_step()
        crew_step.metadata["status"] = "failed"
        crew_step.output = event.error
        crew_step.is_error = True
        await crew_step.update()

    async def _handle_task_started(self, event: TaskStartedUpdate) -> None:
        task_step = await self._ensure_task_step(
            task_id=event.task_id,
            task_name=event.task_name,
            agent_role=event.agent_role,
            content=event.context,
        )
        task_step.metadata["status"] = "running"
        await task_step.update()

    async def _handle_task_completed(self, event: TaskCompletedUpdate) -> None:
        task_step = await self._ensure_task_step(
            task_id=event.task_id,
            task_name=event.task_name,
            agent_role=event.agent_role,
        )
        task_step.metadata["status"] = "completed"
        task_step.output = event.output
        await task_step.update()
        await self._complete_clarification_step(event.task_id)

    async def _handle_task_failed(self, event: TaskFailedUpdate) -> None:
        task_step = await self._ensure_task_step(
            task_id=event.task_id,
            task_name=event.task_name,
            agent_role=event.agent_role,
        )
        task_step.metadata["status"] = "failed"
        task_step.output = event.error
        task_step.is_error = True
        await task_step.update()
        await self._fail_clarification_step(task_id=event.task_id, error=event.error)

    async def _handle_agent_started(self, event: AgentStartedUpdate) -> None:
        if not self._show_agent_steps:
            return
        agent_step = await self._ensure_agent_step(
            task_id=event.task_id,
            task_name=event.task_name,
            agent_key=event.agent_key,
            agent_role=event.agent_role,
            content=event.task_prompt,
            tool_names=event.tool_names,
        )
        agent_step.metadata["status"] = "running"
        await agent_step.update()

    async def _handle_agent_completed(self, event: AgentCompletedUpdate) -> None:
        if not self._show_agent_steps:
            return
        agent_step = await self._ensure_agent_step(
            task_id=event.task_id,
            task_name=event.task_name,
            agent_key=event.agent_key,
            agent_role=event.agent_role,
        )
        agent_step.metadata["status"] = "completed"
        agent_step.output = event.output
        await agent_step.update()

    async def _handle_agent_failed(self, event: AgentFailedUpdate) -> None:
        if not self._show_agent_steps:
            return
        agent_step = await self._ensure_agent_step(
            task_id=event.task_id,
            task_name=event.task_name,
            agent_key=event.agent_key,
            agent_role=event.agent_role,
        )
        agent_step.metadata["status"] = "failed"
        agent_step.output = event.error
        agent_step.is_error = True
        await agent_step.update()

    async def _handle_tool_started(self, event: ToolStartedUpdate) -> None:
        if _is_ask_user_tool(event.tool_name):
            clarification_step = await self._ensure_clarification_step(event)
            question = _extract_tool_question(event.tool_args)
            questions = self._clarification_questions.setdefault(event.task_id or "", [])
            if question and question not in questions:
                questions.append(question)
            clarification_step.metadata["status"] = "waiting_for_user"
            clarification_step.metadata["latest_question"] = question
            clarification_step.output = f"Waiting for your answer to:\n{question}"
            await clarification_step.update()
            return

        tool_step = await self._ensure_tool_step(event)
        tool_step.metadata["status"] = "running"
        await tool_step.update()

    async def _handle_tool_finished(self, event: ToolFinishedUpdate) -> None:
        if _is_ask_user_tool(event.tool_name):
            clarification_step = await self._ensure_clarification_step(event)
            clarification_step.metadata["status"] = "running"
            clarification_step.output = "Collected your answer in chat. Continuing..."
            await clarification_step.update()
            return

        tool_step = await self._ensure_tool_step(event)
        tool_step.metadata["status"] = "completed"
        tool_step.metadata["from_cache"] = event.from_cache
        tool_step.output = event.output
        await tool_step.update()

    async def _handle_tool_failed(self, event: ToolFailedUpdate) -> None:
        if _is_ask_user_tool(event.tool_name):
            clarification_step = await self._ensure_clarification_step(event)
            clarification_step.metadata["status"] = "failed"
            clarification_step.output = event.error
            clarification_step.is_error = True
            await clarification_step.update()
            return

        tool_step = await self._ensure_tool_step(event)
        tool_step.metadata["status"] = "failed"
        tool_step.output = event.error
        tool_step.is_error = True
        await tool_step.update()

    async def _ensure_task_step(
        self,
        task_id: str,
        task_name: str,
        agent_role: str | None,
        content: str | None = None,
    ) -> ChainlitStep:
        existing_step = self._task_steps.get(task_id)
        if existing_step is not None:
            if content and not _stringify(existing_step.input):
                existing_step.input = content
            return existing_step

        task_step = self._cl.Step(
            name=_clean_label(task_name, "Task"),
            type="run",
            parent_id=self._require_crew_step().id,
            show_input="json",
            metadata={
                "component": "task",
                "task_id": task_id,
                "agent_role": agent_role or "",
                "status": "pending",
            },
        )
        if content:
            task_step.input = content
        await task_step.send()
        self._task_steps[task_id] = task_step
        return task_step

    async def _ensure_agent_step(
        self,
        task_id: str,
        task_name: str,
        agent_key: str,
        agent_role: str,
        content: str | None = None,
        tool_names: tuple[str, ...] = (),
    ) -> ChainlitStep:
        step_key = (task_id, agent_key)
        existing_step = self._agent_steps.get(step_key)
        if existing_step is not None:
            if content and not _stringify(existing_step.input):
                existing_step.input = content
            if tool_names:
                existing_step.metadata["tool_names"] = ", ".join(tool_names)
            return existing_step

        task_step = await self._ensure_task_step(
            task_id=task_id,
            task_name=task_name,
            agent_role=agent_role,
        )
        agent_step = self._cl.Step(
            name=_clean_label(agent_role, "Agent"),
            type="run",
            parent_id=task_step.id,
            show_input="json",
            metadata={
                "component": "agent",
                "task_id": task_id,
                "agent_key": agent_key,
                "agent_role": agent_role,
                "tool_names": ", ".join(tool_names),
                "status": "pending",
            },
        )
        if content:
            agent_step.input = content
        await agent_step.send()
        self._agent_steps[step_key] = agent_step
        return agent_step

    async def _ensure_tool_step(
        self,
        event: ToolStartedUpdate | ToolFinishedUpdate | ToolFailedUpdate,
    ) -> ChainlitStep:
        step_key = (
            event.task_id,
            event.agent_key,
            event.tool_name,
            event.run_attempts,
        )
        existing_step = self._tool_steps.get(step_key)
        if existing_step is not None:
            return existing_step

        parent_id = self._require_crew_step().id
        if event.task_id:
            task_step = self._task_steps.get(event.task_id)
            if task_step is not None:
                parent_id = task_step.id

        tool_step = self._cl.Step(
            name=_clean_label(f"Tool: {event.tool_name}", "Tool"),
            type="tool",
            parent_id=parent_id,
            show_input="json",
            default_open=True,
            metadata={
                "component": "tool",
                "task_id": event.task_id or "",
                "task_name": event.task_name or "",
                "agent_key": event.agent_key or "",
                "agent_role": event.agent_role or "",
                "run_attempts": event.run_attempts,
                "status": "pending",
            },
        )
        if isinstance(event, ToolStartedUpdate):
            tool_step.input = event.tool_args
        await tool_step.send()
        self._tool_steps[step_key] = tool_step
        return tool_step

    async def _ensure_clarification_step(
        self,
        event: ToolStartedUpdate | ToolFinishedUpdate | ToolFailedUpdate,
    ) -> ChainlitStep:
        task_id = event.task_id or "clarification"
        existing_step = self._clarification_steps.get(task_id)
        if existing_step is not None:
            return existing_step

        parent_id = self._require_crew_step().id
        if event.task_id:
            task_step = self._task_steps.get(event.task_id)
            if task_step is not None:
                parent_id = task_step.id

        clarification_step = self._cl.Step(
            name="Clarifying Requirements",
            type="run",
            parent_id=parent_id,
            show_input=False,
            default_open=True,
            auto_collapse=False,
            metadata={
                "component": "clarification",
                "task_id": event.task_id or "",
                "task_name": event.task_name or "",
                "agent_key": event.agent_key or "",
                "agent_role": event.agent_role or "",
                "status": "pending",
            },
        )
        await clarification_step.send()
        self._clarification_steps[task_id] = clarification_step
        return clarification_step

    async def _complete_clarification_step(self, task_id: str) -> None:
        clarification_step = self._clarification_steps.get(task_id)
        if clarification_step is None:
            return
        clarification_step.metadata["status"] = "completed"
        questions = self._clarification_questions.get(task_id, [])
        if questions:
            question_summary = "\n".join(f"- {question}" for question in questions)
            clarification_step.output = (
                f"Asked {len(questions)} follow-up question(s) in chat to clarify the request.\n\n"
                f"Questions covered:\n{question_summary}"
            )
        elif not _stringify(clarification_step.output):
            clarification_step.output = "Requirements clarified in chat."
        await clarification_step.update()

    async def _fail_clarification_step(self, task_id: str, error: str) -> None:
        clarification_step = self._clarification_steps.get(task_id)
        if clarification_step is None:
            return
        clarification_step.metadata["status"] = "failed"
        clarification_step.output = error
        clarification_step.is_error = True
        await clarification_step.update()

    def _require_crew_step(self) -> ChainlitStep:
        if self._crew_step is None:
            raise RuntimeError("Crew step renderer was not initialized")
        return self._crew_step
