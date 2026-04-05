import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import ParamSpec, Protocol, TypeAlias, TypeVar

from crewai import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)
from crewai.events.base_events import BaseEvent
from crewai.types.streaming import CrewStreamingOutput

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
    ToolFailedUpdate,
    ToolFinishedUpdate,
    ToolStartedUpdate,
    ToolArguments,
)
from .step_renderer import ChainlitStepRenderer, SupportsChainlitSteps


T_ParamSpec = ParamSpec("T_ParamSpec")
T_Return = TypeVar("T_Return")

CrewKickoffResult: TypeAlias = CrewOutput | CrewStreamingOutput
CrewKickoffInputs: TypeAlias = dict[str, object]
HandlerRegistration: TypeAlias = tuple[type[BaseEvent], Callable[[object, BaseEvent], None]]


class SupportsMakeAsync(Protocol):
    """Protocol for the Chainlit async wrapper used by the adapter."""

    def make_async(
        self,
        function: Callable[T_ParamSpec, T_Return],
        *,
        abandon_on_cancel: bool = False,
        cancellable: bool | None = None,
        limiter: object | None = None,
    ) -> Callable[T_ParamSpec, Awaitable[T_Return]]: ...


class SupportsChainlit(SupportsMakeAsync, SupportsChainlitSteps, Protocol):
    """Protocol for the Chainlit surface used by the adapter."""


class ChainlitCrewAdapter:
    """Async adapter that renders CrewAI execution as Chainlit steps."""

    def __init__(
        self,
        cl: SupportsChainlit,
        crew: Crew,
        inputs: Mapping[str, object] | None = None,
        *,
        show_agent_steps: bool = True,
    ) -> None:
        self.cl = cl
        self.crew = crew
        self.inputs: CrewKickoffInputs = dict(inputs or {})
        self.show_agent_steps = show_agent_steps
        self._crew_task_ids: set[str] = {self._task_id(task) for task in crew.tasks}

    async def kickoff(
        self,
        inputs: Mapping[str, object] | None = None,
    ) -> CrewKickoffResult:
        """Run the crew and mirror its progress into Chainlit steps."""
        if self.crew.stream:
            raise NotImplementedError(
                "ChainlitCrewAdapter step rendering does not support CrewAI streaming crews yet."
            )

        if inputs is not None:
            self.inputs = dict(inputs)

        event_queue: asyncio.Queue[AdapterEvent | None] = asyncio.Queue()
        terminal_event = asyncio.Event()
        renderer = ChainlitStepRenderer(
            cl=self.cl,
            crew_name=self.crew.name,
            inputs=self.inputs,
            show_agent_steps=self.show_agent_steps,
        )
        await renderer.initialize()

        consumer_task = asyncio.create_task(
            self._consume_events(
                event_queue=event_queue,
                renderer=renderer,
                terminal_event=terminal_event,
            )
        )
        loop = asyncio.get_running_loop()
        registrations = self._register_event_handlers(loop, event_queue)

        kickoff_error: BaseException | None = None
        consumer_error: BaseException | None = None

        async_kickoff = self.cl.make_async(self.crew.kickoff)
        try:
            result = await async_kickoff(inputs=self.inputs)
        except BaseException as exc:
            kickoff_error = exc
            result = None
        finally:
            self._unregister_event_handlers(registrations)
            await self._wait_for_terminal_event(
                terminal_event=terminal_event,
                expect_terminal_event=kickoff_error is None,
            )
            await self._wait_for_queue_to_settle(event_queue)
            await event_queue.put(None)
            try:
                await consumer_task
            except BaseException as exc:
                consumer_error = exc

        if kickoff_error is not None:
            raise kickoff_error
        if consumer_error is not None:
            raise consumer_error
        if result is None:
            raise RuntimeError("Crew kickoff finished without a result")
        return result

    async def _consume_events(
        self,
        event_queue: asyncio.Queue[AdapterEvent | None],
        renderer: ChainlitStepRenderer,
        terminal_event: asyncio.Event,
    ) -> None:
        while True:
            event = await event_queue.get()
            if event is None:
                return
            await renderer.handle_event(event)
            if isinstance(event, CrewCompletedUpdate | CrewFailedUpdate):
                terminal_event.set()

    @staticmethod
    async def _wait_for_terminal_event(
        terminal_event: asyncio.Event,
        expect_terminal_event: bool,
    ) -> None:
        if not expect_terminal_event or terminal_event.is_set():
            return
        try:
            await asyncio.wait_for(terminal_event.wait(), timeout=1.0)
        except TimeoutError:
            return

    @staticmethod
    async def _wait_for_queue_to_settle(
        event_queue: asyncio.Queue[AdapterEvent | None],
    ) -> None:
        deadline = asyncio.get_running_loop().time() + 0.2
        while asyncio.get_running_loop().time() < deadline:
            if event_queue.empty():
                await asyncio.sleep(0.02)
                if event_queue.empty():
                    return
            else:
                await asyncio.sleep(0.01)

    def _register_event_handlers(
        self,
        loop: asyncio.AbstractEventLoop,
        event_queue: asyncio.Queue[AdapterEvent | None],
    ) -> list[HandlerRegistration]:
        registrations: list[HandlerRegistration] = []

        def register(
            event_type: type[BaseEvent],
            handler: Callable[[object, BaseEvent], None],
        ) -> None:
            crewai_event_bus.on(event_type)(handler)
            registrations.append((event_type, handler))

        def enqueue(update: AdapterEvent) -> None:
            loop.call_soon_threadsafe(event_queue.put_nowait, update)

        def on_crew_started(source: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, CrewKickoffStartedEvent)
            if source is not self.crew:
                return
            enqueue(
                CrewStartedUpdate(
                    crew_name=typed_event.crew_name or self.crew.name or "Crew",
                    inputs=dict(typed_event.inputs or {}),
                )
            )

        def on_crew_completed(source: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, CrewKickoffCompletedEvent)
            if source is not self.crew:
                return
            enqueue(
                CrewCompletedUpdate(
                    crew_name=typed_event.crew_name or self.crew.name or "Crew",
                    output=self._stringify_output(typed_event.output),
                )
            )

        def on_crew_failed(source: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, CrewKickoffFailedEvent)
            if source is not self.crew:
                return
            enqueue(
                CrewFailedUpdate(
                    crew_name=typed_event.crew_name or self.crew.name or "Crew",
                    error=typed_event.error,
                )
            )

        def on_task_started(source: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, TaskStartedEvent)
            if not self._is_task_for_current_crew(source):
                return
            enqueue(
                TaskStartedUpdate(
                    task_id=self._task_id(source),
                    task_name=self._task_name(source),
                    agent_role=self._task_agent_role(source),
                    context=typed_event.context,
                )
            )

        def on_task_completed(source: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, TaskCompletedEvent)
            if not self._is_task_for_current_crew(source):
                return
            enqueue(
                TaskCompletedUpdate(
                    task_id=self._task_id(source),
                    task_name=self._task_name(source),
                    agent_role=self._task_agent_role(source),
                    output=self._stringify_output(typed_event.output.raw),
                )
            )

        def on_task_failed(source: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, TaskFailedEvent)
            if not self._is_task_for_current_crew(source):
                return
            enqueue(
                TaskFailedUpdate(
                    task_id=self._task_id(source),
                    task_name=self._task_name(source),
                    agent_role=self._task_agent_role(source),
                    error=typed_event.error,
                )
            )

        def on_agent_started(_: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, AgentExecutionStartedEvent)
            if getattr(typed_event.agent, "crew", None) is not self.crew:
                return
            enqueue(
                AgentStartedUpdate(
                    task_id=self._task_id(typed_event.task),
                    task_name=self._task_name(typed_event.task),
                    agent_key=self._agent_key(typed_event.agent),
                    agent_role=typed_event.agent.role,
                    task_prompt=typed_event.task_prompt,
                    tool_names=tuple(tool.name for tool in typed_event.tools or ()),
                )
            )

        def on_agent_completed(_: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, AgentExecutionCompletedEvent)
            if getattr(typed_event.agent, "crew", None) is not self.crew:
                return
            enqueue(
                AgentCompletedUpdate(
                    task_id=self._task_id(typed_event.task),
                    task_name=self._task_name(typed_event.task),
                    agent_key=self._agent_key(typed_event.agent),
                    agent_role=typed_event.agent.role,
                    output=typed_event.output,
                )
            )

        def on_agent_failed(_: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, AgentExecutionErrorEvent)
            if getattr(typed_event.agent, "crew", None) is not self.crew:
                return
            enqueue(
                AgentFailedUpdate(
                    task_id=self._task_id(typed_event.task),
                    task_name=self._task_name(typed_event.task),
                    agent_key=self._agent_key(typed_event.agent),
                    agent_role=typed_event.agent.role,
                    error=typed_event.error,
                )
            )

        def on_tool_started(_: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, ToolUsageStartedEvent)
            if not self._is_tool_for_current_crew(typed_event):
                return
            enqueue(
                ToolStartedUpdate(
                    task_id=typed_event.task_id,
                    task_name=typed_event.task_name,
                    agent_key=self._tool_agent_key(typed_event),
                    agent_role=typed_event.agent_role,
                    tool_name=typed_event.tool_name,
                    tool_args=self._normalize_tool_arguments(typed_event.tool_args),
                    run_attempts=typed_event.run_attempts,
                )
            )

        def on_tool_finished(_: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, ToolUsageFinishedEvent)
            if not self._is_tool_for_current_crew(typed_event):
                return
            enqueue(
                ToolFinishedUpdate(
                    task_id=typed_event.task_id,
                    task_name=typed_event.task_name,
                    agent_key=self._tool_agent_key(typed_event),
                    agent_role=typed_event.agent_role,
                    tool_name=typed_event.tool_name,
                    run_attempts=typed_event.run_attempts,
                    output=self._stringify_output(typed_event.output),
                    from_cache=typed_event.from_cache,
                )
            )

        def on_tool_failed(_: object, event: BaseEvent) -> None:
            typed_event = self._expect_event(event, ToolUsageErrorEvent)
            if not self._is_tool_for_current_crew(typed_event):
                return
            enqueue(
                ToolFailedUpdate(
                    task_id=typed_event.task_id,
                    task_name=typed_event.task_name,
                    agent_key=self._tool_agent_key(typed_event),
                    agent_role=typed_event.agent_role,
                    tool_name=typed_event.tool_name,
                    run_attempts=typed_event.run_attempts,
                    error=self._stringify_output(typed_event.error),
                )
            )

        register(CrewKickoffStartedEvent, on_crew_started)
        register(CrewKickoffCompletedEvent, on_crew_completed)
        register(CrewKickoffFailedEvent, on_crew_failed)
        register(TaskStartedEvent, on_task_started)
        register(TaskCompletedEvent, on_task_completed)
        register(TaskFailedEvent, on_task_failed)
        register(AgentExecutionStartedEvent, on_agent_started)
        register(AgentExecutionCompletedEvent, on_agent_completed)
        register(AgentExecutionErrorEvent, on_agent_failed)
        register(ToolUsageStartedEvent, on_tool_started)
        register(ToolUsageFinishedEvent, on_tool_finished)
        register(ToolUsageErrorEvent, on_tool_failed)

        return registrations

    def _unregister_event_handlers(
        self,
        registrations: list[HandlerRegistration],
    ) -> None:
        for event_type, handler in registrations:
            crewai_event_bus.off(event_type, handler)

    @staticmethod
    def _expect_event(event: BaseEvent, event_type: type[T_Return]) -> T_Return:
        if not isinstance(event, event_type):
            raise TypeError(
                f"Expected event {event_type.__name__}, received {type(event).__name__}"
            )
        return event

    def _is_task_for_current_crew(self, task: object) -> bool:
        agent = getattr(task, "agent", None)
        return getattr(agent, "crew", None) is self.crew

    def _is_tool_for_current_crew(
        self,
        event: ToolUsageStartedEvent | ToolUsageFinishedEvent | ToolUsageErrorEvent,
    ) -> bool:
        if event.task_id is not None and event.task_id in self._crew_task_ids:
            return True
        return False

    @staticmethod
    def _task_id(task: object) -> str:
        return str(getattr(task, "id"))

    @staticmethod
    def _task_name(task: object) -> str:
        name = getattr(task, "name", None)
        if isinstance(name, str) and name.strip():
            return name
        description = getattr(task, "description", "")
        return str(description)

    @staticmethod
    def _task_agent_role(task: object) -> str | None:
        agent = getattr(task, "agent", None)
        role = getattr(agent, "role", None)
        if isinstance(role, str):
            return role
        return None

    @staticmethod
    def _agent_key(agent: object) -> str:
        key = getattr(agent, "key", None)
        if key is not None:
            return str(key)
        agent_id = getattr(agent, "id", None)
        if agent_id is not None:
            return str(agent_id)
        role = getattr(agent, "role", None)
        return str(role or "agent")

    def _tool_agent_key(
        self,
        event: ToolUsageStartedEvent | ToolUsageFinishedEvent | ToolUsageErrorEvent,
    ) -> str | None:
        if event.agent_key is not None:
            return str(event.agent_key)
        if event.agent_id is not None:
            return str(event.agent_id)
        if event.agent is not None:
            return self._agent_key(event.agent)
        return None

    @staticmethod
    def _stringify_output(value: object) -> str:
        if value is None:
            return ""
        raw_value = getattr(value, "raw", None)
        if isinstance(raw_value, str):
            return raw_value
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _normalize_tool_arguments(value: object) -> ToolArguments:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return {str(key): raw_value for key, raw_value in value.items()}
        return str(value)
