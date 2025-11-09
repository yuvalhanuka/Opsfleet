import base64
import contextlib
import io
import logging
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from matplotlib.figure import Figure

from helper_functions import *
from .state import PlotAgentState


class PlotAgent:
    def __init__(self):
        self.script_directory = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        self.system_prompt_dict = self._get_system_prompt_dict()
        self.max_execution_attempts,self.sota_llm_name,self.llm_name = self._get_config()
        self.llm = self._get_llm(model_name = self.llm_name)
        self.sota_llm = self._get_llm(model_name=self.sota_llm_name)
        self.big_query_runner = BigQueryRunner()
        self.df_for_plot = None
        self.plot_fig  = None

    def _get_system_prompt_dict(self):
        system_prompt_dict_path = os.path.join(self.script_directory, 'files','system_prompts.json')
        with open(system_prompt_dict_path, 'rb') as f:
            system_prompt_dict = json.load(f)
        return system_prompt_dict

    def _get_config(self):
        config_path = os.path.join(self.script_directory, 'files','config.json')
        with open(config_path, 'rb') as f:
            config = json.load(f)
        return config['max_execution_attempts'],config['sota_llm_name'],config['llm_name']

    @staticmethod
    def _get_llm(model_name:str):
        load_dotenv()
        return ChatGoogleGenerativeAI(model=model_name)

    def _llm_node_sql_query_generator(self,state:PlotAgentState)-> PlotAgentState:

        sql_query_generator_prompt_template = PromptTemplate(
            input_variables=['tables_information',"recent_attempts","current_time"],
            template=self.system_prompt_dict['sql_query_generator']
        )
        tables_information = get_tables_information()
        current_time = datetime.now().strftime("%Y-%d-%m %H:%M")
        human_msg = HumanMessage('question' + '\n' + state['question'] + '\n' + 'plot description' + state['plot_description'],id="1")
        state['messages'].append(human_msg)

        attempt = 1
        previous_attempts = []
        last_sql = None
        last_error = None

        while attempt <= self.max_execution_attempts:

            attempt_context = "\n\n".join(previous_attempts) if previous_attempts else "None"

            sql_query_generator_prompt = sql_query_generator_prompt_template.format(
                tables_information=tables_information,
                recent_attempts=attempt_context,
                current_time= current_time
            )

            system_msg = SystemMessage(content=sql_query_generator_prompt,id="2")
            state['messages'].append(system_msg)

            response = self.sota_llm.invoke([system_msg,human_msg])
            generated_sql_query = response.content
            state["messages"].append(AIMessage(content=generated_sql_query,id="3"))

            try:
                result_df = self.big_query_runner.execute_query(sql_query=generated_sql_query)
                self.df_for_plot = result_df
                state["messages"].append(AIMessage(content=f"Query: {generated_sql_query}\n execution succeed" ,id="3"))
                break

            except Exception as e:

                err_msg = f"{type(e).__name__}: {e}"

                last_error = err_msg
                last_sql = generated_sql_query
                previous_attempts.append(f"Attempt {attempt} SQL:\n{generated_sql_query}\nError:\n{err_msg}")

                logging.info(f"Execution failed for attempt {attempt}. Error:\n{err_msg}\nFeeding this back to the LLM to try again.")

                attempt += 1



        if self.df_for_plot is None:
            state["messages"].append(AIMessage(
                content=f"Failed to query the SQL database and reached the maximum attempts.\nLast SQL:\n{last_sql}\n\nLast error:\n{last_error}"
                , id="3"
            )
            )

        return state

    def _router_node_check_if_data_fetched(self,state:PlotAgentState)->bool:
        if self.df_for_plot is None:
            return False
        else:
            return True

    def _tool_node_execute_script(self,generated_script):

        try:
            matplotlib.use("Agg")
        except Exception:
            pass


        allowed_builtins = {
            "len": len, "range": range, "min": min, "max": max,
            "sum": sum, "abs": abs, "round": round, "enumerate": enumerate,
            "zip": zip, "print": print
        }


        ns = {
            "__builtins__": allowed_builtins,
            "df": self.df_for_plot,
            "plt": plt,
            "matplotlib": matplotlib,
        }

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(generated_script, ns, ns)

        fig = ns.get("fig", None)
        if fig is None:
            raise ValueError(
                "Generated code did not produce a Figure object named `fig`. "
                "Ensure the script ends with something like: `fig, ax = plt.subplots(...)` and assigns to `fig`."
            )
        if not isinstance(fig, Figure):
            raise TypeError(
                f"Expected a matplotlib.figure.Figure in `fig`, got {type(fig)}. "
                "Ensure you assign the Matplotlib Figure object to `fig`."
            )

        return fig

    def _llm_node_plot_script_generator(self,state:PlotAgentState)-> PlotAgentState:

        plot_script_generator_prompt_template = PromptTemplate(
            input_variables=['dataframe_columns',"recent_attempts","df_shape","df_sample_rows","sql_query_used"],
            template=self.system_prompt_dict['plot_script_generator']
        )

        dataframe_columns = self.df_for_plot.columns
        df_shape = self.df_for_plot.shape
        df_sample_rows = self.df_for_plot.head(1).to_string()
        sql_query_used = state['messages'][-1]
        human_msg = HumanMessage('question' + '\n' + state['question'] + '\n' + 'plot description' + state['plot_description'], id="1")

        attempt = 1
        previous_attempts = []
        last_script = None
        last_error = None

        while attempt <= self.max_execution_attempts:
            attempt_context = "\n\n".join(previous_attempts) if previous_attempts else "None"

            plot_script_generator_prompt = plot_script_generator_prompt_template.format(
                dataframe_columns=dataframe_columns,
                recent_attempts=attempt_context,
                df_shape=df_shape,
                df_sample_rows = df_sample_rows,
                sql_query_used = sql_query_used
            )
            system_msg = SystemMessage(content=plot_script_generator_prompt,id="2")
            state['messages'].append(system_msg)
            response = self.sota_llm.invoke([system_msg,human_msg])
            generated_script = response.content
            state["messages"].append(AIMessage(content=generated_script,id="3"))

            try:

                self.plot_fig = self._tool_node_execute_script(generated_script=generated_script)
                state["messages"].append(AIMessage(content=f"Script: {generated_script}\n execution succeed" ,id="3"))
                break

            except Exception as e:

                err_msg = f"{type(e).__name__}: {e}"

                last_error = err_msg
                last_script = generated_script

                previous_attempts.append(f"Attempt {attempt} Script:\n{generated_script}\nError:\n{err_msg}")
                logging.info(f"Execution failed for attempt {attempt}. Error:\n{err_msg}\nFeeding this back to the LLM to try again.")

                attempt += 1

        if self.plot_fig is None:
            state["messages"].append(AIMessage(
                content=f"Failed to generate script for plot and reached the maximum attempts.\nLast SQL:\n{last_script}\n\nLast error:\n{last_error}"
                , id="3"
            )
            )
        return state

    def _router_node_check_if_plot_generated(self, state: PlotAgentState) -> bool:
        if self.plot_fig is None:
            return False
        else:
            return True

    def _llm_node_plot_analysis_generator(self,state:PlotAgentState)-> PlotAgentState:

        plot_analysis_generator_prompt = self.system_prompt_dict['plot_analysis_generator']
        system_msg = SystemMessage(content=plot_analysis_generator_prompt, id="2")
        state['messages'].append(system_msg)

        plots_dir = os.path.join(self.script_directory, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        file_name = state['question']+'.png'
        saved_plot_path = os.path.join(plots_dir, file_name)
        self.plot_fig.savefig(saved_plot_path, format="png", bbox_inches="tight", dpi=144)


        buf = io.BytesIO()
        self.plot_fig.savefig(buf, format="png", bbox_inches="tight", dpi=144)
        buf.seek(0)
        b64_png = base64.b64encode(buf.read()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_png}"
        text_input = 'question' + '\n' + state['question'] + '\n\n' + 'plot description' + state['plot_description']

        human_msg = HumanMessage(content=[
            {"type": "text", "text": text_input},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])

        response = self.llm.invoke([system_msg, human_msg])
        plot_analysis = response.content

        state["messages"].append(AIMessage(content=plot_analysis, id="3"))
        state["plot_analysis"] = plot_analysis
        state['saved_plot_path'] = saved_plot_path

        return state

    def _llm_node_error_explainer(self,state:PlotAgentState)-> PlotAgentState:

        error_explainer_prompt = self.system_prompt_dict['error_explainer']
        system_msg = SystemMessage(content=error_explainer_prompt, id="2")

        ai_msg = AIMessage(content= state['messages'][-1])

        state['messages'].append(system_msg)


        response = self.llm.invoke([system_msg,ai_msg])
        answer = response.content

        state["messages"].append(AIMessage(content=answer, id="3"))
        state["plot_analysis"] = answer
        state['saved_plot_path'] = None

        return state

    def get_plot_agent(self)->CompiledStateGraph:

        builder = StateGraph(PlotAgentState)

        builder.add_node('SQL query generator',self._llm_node_sql_query_generator)
        builder.add_node('Plot script generator',self._llm_node_plot_script_generator)
        builder.add_node('Plot analysis generator',self._llm_node_plot_analysis_generator)
        builder.add_node('Error explainer',self._llm_node_error_explainer)

        builder.add_edge(START, 'SQL query generator')

        builder.add_conditional_edges(
            source="SQL query generator",
            path=self._router_node_check_if_data_fetched,
            path_map={
                    True:'Plot script generator',
                    False:'Error explainer'
            }
        )

        builder.add_conditional_edges(
            source="Plot script generator",
            path=self._router_node_check_if_plot_generated,
            path_map={
                    True:'Plot analysis generator',
                    False:'Error explainer'
            }
        )

        builder.add_edge('Error explainer',END)
        builder.add_edge('Plot analysis generator', END)

        return builder.compile()


