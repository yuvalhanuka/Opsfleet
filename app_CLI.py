import os
import sys
import textwrap
import warnings
import traceback

warnings.filterwarnings("ignore")
from helper_functions import *
from data_analysis_agent import DataAnalysisAgent, DataAnalysisAgentState
import os



def print_header():
    bar = "=" * 72
    print(bar)
    print(" Data Analysis CLI ".center(72, "="))
    print(bar)
    print("Ask a question about the ecommerce data (type 'exit' to quit).")
    print()

def pretty_print_answer(answer: str | None):
    if not answer:
        return
    print("\nAnswer:\n" + "-" * 72)
    print(textwrap.fill(str(answer), width=100))
    print("-" * 72)

def show_plot_if_any(plot_path: str | None):
    if not plot_path or not os.path.exists(plot_path):
        return
    try:
        from PIL import Image
        Image.open(plot_path).show()
    except Exception:
        try:
            if sys.platform.startswith("darwin"):
                os.system(f"open '{plot_path}'")
            elif os.name == "nt":
                os.startfile(plot_path)
            else:
                os.system(f"xdg-open '{plot_path}' >/dev/null 2>&1 &")
        except Exception:
            pass

def main():
    setup_logging()
    print_header()

    app = DataAnalysisAgent().get_data_analysis_agent()

    memory_messages = []

    while True:
        try:
            user_q = input("\n> Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_q:
            continue

        if user_q.lower() in {"quit", "q", "exit"}:
            print("Goodbye!")
            break

        state: DataAnalysisAgentState = {
            "user_question": user_q,
            "messages": memory_messages,
        }

        try:
            final_state = app.invoke(state)
        except Exception as e:
            full_trace = "".join(traceback.format_exception(e))
            print("\n[Error] Sorry—something went wrong processing your question.\n")
            logging.error(full_trace)
            continue

        memory_messages = final_state.get("messages")

        answer = final_state.get("chat_response")
        pretty_print_answer(answer)

        plot_path = final_state.get("plot_file_path")
        show_plot_if_any(plot_path)

if __name__ == "__main__":
    main()
