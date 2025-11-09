from typing import TypedDict


class SqlAgentState(TypedDict):
    question: str
    messages: list
    response: str