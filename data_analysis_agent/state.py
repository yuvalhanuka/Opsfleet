from typing import List, TypedDict, Literal, Optional

from pydantic import BaseModel, Field


class SupervisorOutput(BaseModel):
    response: str = Field(
        ...,
        description=(
            "The direct chat reply to the user. "
            "Must be non-empty if decision == 'response'. "
            "Must be empty if decision == 'explore'."
        )
    )
    explore: str = Field(
        ...,
        description=(
            "A concise description of what the user wants (intent/summary). "
            "Must be non-empty if decision == 'explore'. "
            "Must be empty if decision == 'response'."
        )
    )
    decision: Literal['explore','response'] = Field(
        ...,
        description=(
            "Supervisor choice. "
            "'response' → answer now . "
            "'explore' → gather more information via SQL, plots agents."
        )
    )

class ExplorerOutput(BaseModel):
    questions_for_sql_agent: List[str] = Field(
        ...,
        description=(
            "A list of precise, self-contained questions for the SQL agent, derived from the user’s intent. "
            "Leave as an empty list if no database retrieval is needed."
        )
    )
    plot_description: str = Field(
        ...,
        description=(
            "A concise, detailed description of the plot to generate (x-axis, y-axis, series/grouping, labels, title, "
            "sorting, filters/aggregation). Leave empty if no plot is needed."
        )
    )

class DataAnalysisAgentState(TypedDict):
    user_question:str
    messages :list
    supervisor_decision : SupervisorOutput
    explorer_decision: ExplorerOutput
    sql_agent_response: List[dict]
    plot_agent_response: Optional[str]
    plot_file_path: Optional[str]
    chat_response:str