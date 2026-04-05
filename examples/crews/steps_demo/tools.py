from __future__ import annotations

import json
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from pydantic import BaseModel, Field

from crewai.tools import BaseTool


JSONScalar = None | bool | int | float | str
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

BASE_URL = "https://jsonplaceholder.typicode.com"


def _fetch_json(path: str) -> JSONValue:
    url = f"{BASE_URL}/{path.lstrip('/')}"
    try:
        with urlopen(url, timeout=15) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        return {"error": f"HTTP error {exc.code} while requesting {url}"}
    except URLError as exc:
        return {"error": f"Network error while requesting {url}: {exc.reason}"}

    parsed = cast(JSONValue, json.loads(payload))
    return parsed


def _format_json(path: str, data: JSONValue) -> str:
    return f"Endpoint: /{path.lstrip('/')}\n{json.dumps(data, indent=2)}"


class FetchPostInput(BaseModel):
    """Input schema for fetching a post."""

    post_id: int = Field(
        ...,
        description="The JSONPlaceholder post id to retrieve.",
        ge=1,
        le=100,
    )


class FetchPostTool(BaseTool):
    name: str = "Fetch post from JSONPlaceholder"
    description: str = (
        "Fetch a single post from JSONPlaceholder using a post id. Use this when you "
        "need the post title, body, and author userId."
    )
    args_schema: type[BaseModel] = FetchPostInput

    def _run(self, post_id: int) -> str:
        path = f"posts/{post_id}"
        return _format_json(path, _fetch_json(path))


class FetchUserInput(BaseModel):
    """Input schema for fetching a user."""

    user_id: int = Field(
        ...,
        description="The JSONPlaceholder user id to retrieve.",
        ge=1,
        le=10,
    )


class FetchUserTool(BaseTool):
    name: str = "Fetch user from JSONPlaceholder"
    description: str = (
        "Fetch a single user profile from JSONPlaceholder using a user id. Use this "
        "for username, email, company, and address details."
    )
    args_schema: type[BaseModel] = FetchUserInput

    def _run(self, user_id: int) -> str:
        path = f"users/{user_id}"
        return _format_json(path, _fetch_json(path))


class FetchTodosInput(BaseModel):
    """Input schema for fetching todos."""

    user_id: int = Field(
        ...,
        description="The JSONPlaceholder user id whose todos should be retrieved.",
        ge=1,
        le=10,
    )
    limit: int = Field(
        default=5,
        description="Maximum number of todos to include in the result.",
        ge=1,
        le=20,
    )


class FetchTodosTool(BaseTool):
    name: str = "Fetch user todos from JSONPlaceholder"
    description: str = (
        "Fetch todos for a JSONPlaceholder user. Use this to understand the user's "
        "recent todo items and completion status."
    )
    args_schema: type[BaseModel] = FetchTodosInput

    def _run(self, user_id: int, limit: int = 5) -> str:
        path = f"todos?userId={user_id}"
        data = _fetch_json(path)
        if isinstance(data, list):
            limited_data: JSONValue = data[:limit]
        else:
            limited_data = data
        return _format_json(path, limited_data)


class FetchCommentsInput(BaseModel):
    """Input schema for fetching comments."""

    post_id: int = Field(
        ...,
        description="The JSONPlaceholder post id whose comments should be retrieved.",
        ge=1,
        le=100,
    )
    limit: int = Field(
        default=3,
        description="Maximum number of comments to include in the result.",
        ge=1,
        le=20,
    )


class FetchCommentsTool(BaseTool):
    name: str = "Fetch post comments from JSONPlaceholder"
    description: str = (
        "Fetch comments for a JSONPlaceholder post. Use this to understand how readers "
        "reacted to the post."
    )
    args_schema: type[BaseModel] = FetchCommentsInput

    def _run(self, post_id: int, limit: int = 3) -> str:
        path = f"comments?postId={post_id}"
        data = _fetch_json(path)
        if isinstance(data, list):
            limited_data: JSONValue = data[:limit]
        else:
            limited_data = data
        return _format_json(path, limited_data)


def build_jsonplaceholder_tools() -> tuple[
    FetchPostTool,
    FetchUserTool,
    FetchTodosTool,
    FetchCommentsTool,
]:
    """Create the reusable JSONPlaceholder demo tools."""
    return (
        FetchPostTool(),
        FetchUserTool(),
        FetchTodosTool(),
        FetchCommentsTool(),
    )
