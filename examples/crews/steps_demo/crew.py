import os

from crewai import Agent, Crew, LLM, Process, Task

from .tools import build_jsonplaceholder_tools


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def build_steps_demo_crew() -> Crew:
    """Create a demo crew that exercises task, agent, and tool step rendering."""
    llm = LLM(model=DEFAULT_MODEL, temperature=0.1)
    fetch_post_tool, fetch_user_tool, fetch_todos_tool, fetch_comments_tool = (
        build_jsonplaceholder_tools()
    )

    post_analyst = Agent(
        role="Post Analyst",
        goal=(
            "Fetch the requested JSONPlaceholder post and identify the author details "
            "needed by the rest of the crew."
        ),
        backstory=(
            "You are careful with API-driven analysis. You always fetch the actual post "
            "before summarizing it."
        ),
        llm=llm,
        tools=[fetch_post_tool],
        verbose=True,
    )

    author_analyst = Agent(
        role="Author Analyst",
        goal=(
            "Fetch the post author's profile and todos so the crew can describe the "
            "author with real API data."
        ),
        backstory=(
            "You are skilled at connecting post data with author context. You use the "
            "available API tools to ground your findings."
        ),
        llm=llm,
        tools=[fetch_user_tool, fetch_todos_tool],
        verbose=True,
    )

    report_writer = Agent(
        role="Report Writer",
        goal=(
            "Fetch post comments and write a concise final report that combines the "
            "post, author, and comment insights."
        ),
        backstory=(
            "You turn API findings into clear summaries. You always inspect comments "
            "before delivering the final report."
        ),
        llm=llm,
        tools=[fetch_comments_tool],
        verbose=True,
    )

    post_task = Task(
        name="Analyze Post",
        description=(
            "Analyze JSONPlaceholder post `{post_id}`.\n"
            "You must call the `Fetch post from JSONPlaceholder` tool with that exact "
            "post id before writing your answer.\n"
            "Return:\n"
            "- the post title\n"
            "- a 1-2 sentence summary of the post body\n"
            "- the numeric `userId` for the post author"
        ),
        expected_output=(
            "A short note containing the post title, summary, and the author userId."
        ),
        agent=post_analyst,
    )

    author_task = Task(
        name="Analyze Author",
        description=(
            "Using the previous task output, identify the post author's `userId`.\n"
            "Then you must call both:\n"
            "- `Fetch user from JSONPlaceholder`\n"
            "- `Fetch user todos from JSONPlaceholder`\n"
            "Return:\n"
            "- the author's name and username\n"
            "- the author's email and company\n"
            "- a short summary of the first few todos and how many are completed"
        ),
        expected_output=(
            "A concise author briefing with profile details and todo insights."
        ),
        agent=author_analyst,
        context=[post_task],
    )

    report_task = Task(
        name="Write Report",
        description=(
            "Using the prior task outputs, you must call `Fetch post comments from "
            "JSONPlaceholder` for post `{post_id}` before writing the final answer.\n"
            "Write a user-facing report that includes:\n"
            "- which post id was analyzed\n"
            "- the post summary\n"
            "- who the author is\n"
            "- what the author's todos suggest\n"
            "- a brief comment summary based on the fetched comments"
        ),
        expected_output="A polished final report for the user.",
        agent=report_writer,
        context=[post_task, author_task],
    )

    return Crew(
        name="Step Rendering Demo Crew",
        agents=[post_analyst, author_analyst, report_writer],
        tasks=[post_task, author_task, report_task],
        process=Process.sequential,
        verbose=True,
    )
