# chainlit-crew-adapter

`chainlit-crew-adapter` is a small library for running CrewAI crews inside Chainlit with a cleaner developer experience.

It focuses on two things:

- rendering CrewAI execution as Chainlit steps
- letting CrewAI agents ask the user follow-up questions with a reusable Chainlit-backed tool

## Demo

![chainlit-crew-adapter demo](docs/assests/chainlit-crew-adapter.gif)

## Features

- `ChainlitCrewAdapter` with async `kickoff(...)`
- step rendering for crew, task, agent, and tool events
- optional `show_agent_steps=False` for cleaner traces
- `ChainlitAskUserTool` for human-in-the-loop follow-up questions
- examples for both step rendering and ask-user workflows

## Installation

### Local development

```bash
pip install -e .
```

Or, if you prefer the repo requirements file:

```bash
pip install -r requirements.txt
```

### Environment variables

At minimum, set:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_MODEL` is optional. The examples default to `gpt-4o-mini`.

## Quick Start

### Step rendering only

```python
import chainlit as cl
from chainlit_crew_adapter import ChainlitCrewAdapter

adapter = ChainlitCrewAdapter(cl=cl, crew=crew)
result = await adapter.kickoff(inputs={"user_request": message.content})
```

### Human follow-up questions

```python
import chainlit as cl
from chainlit_crew_adapter import ChainlitAskUserTool, ChainlitCrewAdapter

ask_user_tool = ChainlitAskUserTool(timeout_seconds=180, author="Crew Assistant")
crew = build_ask_user_demo_crew(ask_user_tool=ask_user_tool)
adapter = ChainlitCrewAdapter(cl=cl, crew=crew, show_agent_steps=False)
result = await adapter.kickoff(inputs={"user_request": message.content})
```

## API

### `ChainlitCrewAdapter`

```python
ChainlitCrewAdapter(
    cl: SupportsChainlit,
    crew: Crew,
    inputs: Mapping[str, object] | None = None,
    *,
    show_agent_steps: bool = True,
)
```

#### `await adapter.kickoff(inputs=None)`

Runs `crew.kickoff(...)` through Chainlit and mirrors CrewAI progress into the Chainlit step tree.

### `ChainlitAskUserTool`

```python
ChainlitAskUserTool(
    *,
    ask_user=None,
    chainlit_context=None,
    timeout_seconds: int = 120,
    author: str = "Crew Assistant",
    fail_on_timeout: bool = False,
)
```

Use this tool in CrewAI agents when they need to ask the user follow-up questions during execution.

## Example Apps

This repo includes two example apps.

### 1. Step rendering demo

File:

```bash
examples/steps_demo.py
```

Run it with:

```bash
chainlit run examples/steps_demo.py
```

Try prompts like:

```text
Analyze post 5
```

```text
Summarize post 12
```

What it demonstrates:

- crew step rendering
- task, agent, and tool steps
- a real tool-driven CrewAI workflow with JSONPlaceholder

### 2. Ask-user demo

File:

```bash
examples/ask_user_demo.py
```

Run it with:

```bash
chainlit run examples/ask_user_demo.py
```

Try prompts like:

```text
Help me plan a product launch
```

```text
I need a landing page brief
```

What it demonstrates:

- `ChainlitAskUserTool`
- chat-based follow-up questions during crew execution
- a compact step trace using `show_agent_steps=False`
- a dedicated clarification step summarizing the questions asked in chat

### Legacy alias

For backward compatibility, this file points to the ask-user demo:

```bash
examples/manual_crew_integration.py
```

## Example Structure

```text
chainlit_crew_adapter/
├── chainlit_crew_adapter/
│   ├── __init__.py
│   ├── adapter.py
│   ├── events.py
│   ├── human_input.py
│   └── step_renderer.py
├── examples/
│   ├── ask_user_demo.py
│   ├── manual_crew_integration.py
│   ├── steps_demo.py
│   └── crews/
│       ├── ask_user_demo/
│       └── steps_demo/
├── pyproject.toml
└── README.md
```

## Notes

- CrewAI streaming crews are not supported yet in the adapter step renderer.
- The ask-user demo intentionally keeps the back-and-forth in chat and the execution trace in steps.
- The step renderer treats the ask-user tool as a special clarification flow to avoid noisy duplicate traces.

## Development Checks

A quick local verification pass:

```bash
python -m py_compile chainlit_crew_adapter/*.py examples/*.py examples/crews/*/*.py
```

## Next Step Toward Deployment

The repo now has a basic `pyproject.toml`, package data marker, examples, and a cleaner separation between library code and demos. The next natural step is publishing metadata cleanup and then building the distribution with:

```bash
python -m build
```
