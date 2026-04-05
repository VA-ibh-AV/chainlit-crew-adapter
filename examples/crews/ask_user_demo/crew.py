import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def build_ask_user_demo_crew(*, ask_user_tool: BaseTool) -> Crew:
    """Create a demo crew that asks the user follow-up questions mid-run."""
    llm = LLM(model=DEFAULT_MODEL, temperature=0.1)

    intake_specialist = Agent(
        role="Intake Specialist",
        goal=(
            "Collect the missing details needed to turn the user's request into a "
            "clear, actionable brief."
        ),
        backstory=(
            "You are a careful project intake specialist. You ask focused follow-up "
            "questions one at a time and make sure the rest of the crew receives a "
            "complete brief."
        ),
        llm=llm,
        tools=[ask_user_tool],
        verbose=True,
    )

    solution_planner = Agent(
        role="Solution Planner",
        goal=(
            "Turn the clarified intake details into a practical execution plan with "
            "clear priorities and next steps."
        ),
        backstory=(
            "You are strong at turning messy requests into practical plans. You keep "
            "the plan grounded in the user's real constraints."
        ),
        llm=llm,
        verbose=True,
    )

    response_writer = Agent(
        role="Response Writer",
        goal=(
            "Write a polished final response that feels helpful, concrete, and easy "
            "for the user to act on."
        ),
        backstory=(
            "You are an excellent collaborator who turns planning notes into clear, "
            "friendly guidance."
        ),
        llm=llm,
        verbose=True,
    )

    intake_task = Task(
        name="Clarify Requirements",
        description=(
            "The user's starting request is: {user_request}\n"
            "Before you write your answer, you must call `Ask user a follow-up question` "
            "at least twice. Ask one question at a time.\n"
            "You must gather at least these details:\n"
            "- the main goal or desired outcome\n"
            "- the timeline or deadline\n"
            "- any must-have constraints or preferences\n"
            "After asking your follow-up questions, return an intake brief with:\n"
            "- a restated request\n"
            "- clarified goal\n"
            "- timeline\n"
            "- constraints or preferences\n"
            "- any open questions that still remain"
        ),
        expected_output=(
            "A concise intake brief with clarified goals, timing, constraints, and "
            "remaining unknowns."
        ),
        agent=intake_specialist,
    )

    planning_task = Task(
        name="Create Plan",
        description=(
            "Using the intake brief from the prior task, create a practical plan for "
            "the user.\n"
            "Return:\n"
            "- the top objective\n"
            "- 3 to 5 recommended workstreams or phases\n"
            "- the most important tradeoffs or risks\n"
            "- the immediate next actions the user should take"
        ),
        expected_output=(
            "A practical plan with priorities, workstreams, risks, and next actions."
        ),
        agent=solution_planner,
        context=[intake_task],
    )

    response_task = Task(
        name="Write Final Response",
        description=(
            "Using the intake brief and the plan, write a polished user-facing response.\n"
            "The response should include:\n"
            "- a short summary of what the user is trying to achieve\n"
            "- the clarified details gathered from the user\n"
            "- the recommended plan\n"
            "- a short section called `Next Steps`"
        ),
        expected_output="A polished final response for the user.",
        agent=response_writer,
        context=[intake_task, planning_task],
    )

    return Crew(
        name="Ask User Demo Crew",
        agents=[intake_specialist, solution_planner, response_writer],
        tasks=[intake_task, planning_task, response_task],
        process=Process.sequential,
        verbose=True,
    )
