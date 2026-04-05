# Chainlit Crew Adapter Approach

## Goal

Build a `chainlit-crew-adapter` library that makes it easy for developers to:

1. run a CrewAI `crew` from Chainlit,
2. pass `inputs` into `crew.kickoff(...)`, and
3. show what the crew is doing as Chainlit steps.

## Core Problem

CrewAI execution details are not natively rendered as Chainlit steps.

CrewAI does emit rich execution events, including:

- crew lifecycle events,
- task lifecycle events,
- agent lifecycle events,
- tool lifecycle events.

These can be used to drive the Chainlit UI.

## Key Insight

We should use the CrewAI event system as the source of truth for progress and execution state.

Relevant CrewAI capabilities:

- `BaseEventListener`
- `crewai_event_bus.on(...)`
- `crewai_event_bus.off(...)`
- crew, task, agent, and tool event types

## Important Constraint

CrewAI event handlers run through CrewAI's global event bus and are not guaranteed to run on the active Chainlit request loop.

Because of that, the adapter should **not** directly call Chainlit UI APIs from CrewAI event handlers.

Instead, the adapter should:

1. listen to CrewAI events,
2. normalize them into adapter events,
3. push them into an internal queue,
4. consume that queue from the Chainlit async context,
5. create and update `cl.Step` objects there.

## Recommended Architecture

### 1. Adapter

Public API:

```python
adapter = ChainlitCrewAdapter(cl=cl, crew=crew)
result = await adapter.kickoff(inputs={"question": message.content})
```

Responsibilities:

- accept `cl`, `crew`, and `inputs`,
- register temporary CrewAI event handlers,
- run `crew.kickoff(...)`,
- start and stop the event consumer,
- return the final crew result.

### 2. Event Bridge

Convert raw CrewAI events into internal adapter events such as:

- `crew_started`
- `crew_completed`
- `crew_failed`
- `task_started`
- `task_completed`
- `task_failed`
- `agent_started`
- `agent_completed`
- `agent_failed`
- `tool_started`
- `tool_finished`
- `tool_failed`

This layer should stay independent from Chainlit UI concerns.

### 3. Chainlit Renderer

Consume normalized adapter events and render them into Chainlit steps.

Recommended step hierarchy:

- Crew step
- Task step
- Agent step
- Tool step

This gives a clear nested view of execution without exposing raw event internals to the app developer.

## Why Not Use `scoped_handlers()` As The Main Strategy

CrewAI provides `crewai_event_bus.scoped_handlers()`, but it is better suited for temporary or test-only usage.

For a Chainlit adapter, the safer production approach is:

1. register only the adapter's handlers,
2. keep references to them,
3. unregister them explicitly in `finally` using `crewai_event_bus.off(...)`.

Reason:

- the CrewAI event bus is global,
- removing all handlers during a request can be risky in a multi-session app,
- explicit registration and cleanup gives more predictable behavior.

## Event Filtering

Since the CrewAI event bus is global, the adapter must ignore events that do not belong to the current `crew`.

Suggested filtering rules:

### Crew events

Use only events where:

```python
source is self.crew
```

### Task events

Use only events where:

```python
getattr(source, "agent", None) is not None
and source.agent.crew is self.crew
```

### Agent events

Use only events where:

```python
event.agent.crew is self.crew
```

### Tool events

Use only events where:

```python
event.agent is not None
and event.agent.crew is self.crew
```

## Step Correlation Strategy

To update the right Chainlit step when a completion or failure event arrives, the adapter should keep registries such as:

- `task_steps[task_id] -> cl.Step`
- `agent_steps[(task_id, agent_key)] -> cl.Step`
- `tool_steps[(task_id, agent_id, tool_name, run_attempts)] -> cl.Step`

This allows the adapter to:

- create a step on `started`,
- update the same step on `completed`,
- mark it as error on `failed`.

## `kickoff()` Flow

Recommended internal flow for `ChainlitCrewAdapter.kickoff(...)`:

1. capture the current Chainlit/asyncio context,
2. create an internal queue for adapter events,
3. create the root crew step,
4. register CrewAI event handlers,
5. start the queue consumer task,
6. run `await cl.make_async(crew.kickoff)(inputs=inputs)`,
7. flush pending CrewAI event handlers,
8. unregister adapter handlers,
9. stop the queue consumer,
10. return the crew result.

## Why Queue-Based Bridging Is Safer

Benefits:

- keeps CrewAI event processing separate from Chainlit rendering,
- avoids unsafe direct UI updates from CrewAI handler threads,
- makes testing easier,
- gives us one place to normalize and filter events,
- supports future renderers beyond Chainlit if needed.

## Suggested V1 Scope

Implement only these event groups first:

- crew start / complete / fail
- task start / complete / fail
- agent start / complete / fail
- tool start / finish / fail

Do **not** include these in V1:

- token streaming,
- memory events,
- knowledge events,
- training/test events,
- tracing integrations.

This keeps the first implementation focused and stable.

## Future Enhancements

Possible next steps after V1:

1. LLM streaming support using Chainlit token streaming
2. configurable visibility
   - show/hide task steps
   - show/hide agent steps
   - show/hide tool steps
3. callback hooks for custom rendering
4. support for custom event transformers
5. richer metadata on steps
6. replay/debug mode

## Proposed Implementation Order

1. define normalized internal adapter event models,
2. implement per-kickoff event registration and cleanup,
3. build queue bridge from CrewAI events to Chainlit loop,
4. build Chainlit step renderer,
5. integrate with the example app,
6. refine typing and extensibility.

## Summary

The adapter should be designed as an event bridge, not just a thin async wrapper.

The best path is:

- listen to CrewAI events,
- filter only the current crew's execution,
- queue normalized events,
- render them as Chainlit steps from the active async context.

This gives developers a simple API while making CrewAI progress visible in Chainlit.
