from typing import TypedDict, Optional


class PlotAgentState(TypedDict):
    question: str
    plot_description: list
    messages: list
    plot_analysis: str
    saved_plot_path : Optional[str]