from dataclasses import dataclass
from typing import TypeAlias


ToolArguments: TypeAlias = dict[str, object] | str


@dataclass(slots=True, frozen=True)
class CrewStartedUpdate:
    crew_name: str
    inputs: dict[str, object]


@dataclass(slots=True, frozen=True)
class CrewCompletedUpdate:
    crew_name: str
    output: str


@dataclass(slots=True, frozen=True)
class CrewFailedUpdate:
    crew_name: str
    error: str


@dataclass(slots=True, frozen=True)
class TaskStartedUpdate:
    task_id: str
    task_name: str
    agent_role: str | None
    context: str | None


@dataclass(slots=True, frozen=True)
class TaskCompletedUpdate:
    task_id: str
    task_name: str
    agent_role: str | None
    output: str


@dataclass(slots=True, frozen=True)
class TaskFailedUpdate:
    task_id: str
    task_name: str
    agent_role: str | None
    error: str


@dataclass(slots=True, frozen=True)
class AgentStartedUpdate:
    task_id: str
    task_name: str
    agent_key: str
    agent_role: str
    task_prompt: str
    tool_names: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class AgentCompletedUpdate:
    task_id: str
    task_name: str
    agent_key: str
    agent_role: str
    output: str


@dataclass(slots=True, frozen=True)
class AgentFailedUpdate:
    task_id: str
    task_name: str
    agent_key: str
    agent_role: str
    error: str


@dataclass(slots=True, frozen=True)
class ToolStartedUpdate:
    task_id: str | None
    task_name: str | None
    agent_key: str | None
    agent_role: str | None
    tool_name: str
    tool_args: ToolArguments
    run_attempts: int


@dataclass(slots=True, frozen=True)
class ToolFinishedUpdate:
    task_id: str | None
    task_name: str | None
    agent_key: str | None
    agent_role: str | None
    tool_name: str
    run_attempts: int
    output: str
    from_cache: bool


@dataclass(slots=True, frozen=True)
class ToolFailedUpdate:
    task_id: str | None
    task_name: str | None
    agent_key: str | None
    agent_role: str | None
    tool_name: str
    run_attempts: int
    error: str


AdapterEvent: TypeAlias = (
    CrewStartedUpdate
    | CrewCompletedUpdate
    | CrewFailedUpdate
    | TaskStartedUpdate
    | TaskCompletedUpdate
    | TaskFailedUpdate
    | AgentStartedUpdate
    | AgentCompletedUpdate
    | AgentFailedUpdate
    | ToolStartedUpdate
    | ToolFinishedUpdate
    | ToolFailedUpdate
)
