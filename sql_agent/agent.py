import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from helper_functions import *
from .state import SqlAgentState


class SqlAgent:
    def __init__(self):
        self.script_directory = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        self.system_prompt_dict = self._get_system_prompt_dict()
        self.max_execution_attempts,self.sota_llm_name,self.llm_name = self._get_config()
        self.llm = self._get_llm(model_name = self.llm_name)
        self.sota_llm = self._get_llm(model_name=self.sota_llm_name)
        self.big_query_runner = BigQueryRunner()

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

    def _llm_node_sql_query_generator(self,state:SqlAgentState)-> SqlAgentState:

        sql_query_generator_prompt_template = PromptTemplate(
            input_variables=['tables_information',"recent_attempts","current_time"],
            template=self.system_prompt_dict['sql_query_generator']
        )
        tables_information = get_tables_information()
        current_time = datetime.now().strftime("%Y-%d-%m %H:%M")
        human_msg = HumanMessage(state['question'],id="1")
        state['messages'].append(human_msg)

        attempt = 1
        previous_attempts = []
        execution_result = None
        last_sql = None
        last_error = None

        while attempt <= self.max_execution_attempts:

            attempt_context = "\n\n".join(previous_attempts) if previous_attempts else "None"

            sql_query_generator_prompt = sql_query_generator_prompt_template.format(
                tables_information=tables_information,
                recent_attempts=attempt_context,
                current_time = current_time
            )

            system_msg = SystemMessage(content=sql_query_generator_prompt,id="2")
            state['messages'].append(system_msg)

            response = self.sota_llm.invoke([system_msg,human_msg])
            generated_sql_query = response.content
            state["messages"].append(AIMessage(content=generated_sql_query,id="3"))

            try:
                result_df = self.big_query_runner.execute_query(sql_query=generated_sql_query)
                execution_result = result_df.head(100).to_string(index=False)
                state["messages"].append(AIMessage(content=f"Query: {generated_sql_query}\n Query execution result :\n {execution_result}",id="3"))
                break

            except Exception as e:

                err_msg = f"{type(e).__name__}: {e}"

                last_error = err_msg
                last_sql = generated_sql_query
                previous_attempts.append(f"Attempt {attempt} SQL:\n{generated_sql_query}\nError:\n{err_msg}")

                logging.error(f" SQl agent | Execution failed for attempt {attempt}. | Error:\n{err_msg} ")

                attempt += 1



        if execution_result is None:
            state["messages"].append(AIMessage(
                content=f"Failed to query the SQL database and reached the maximum attempts.\nLast SQL:\n{last_sql}\n\nLast error:\n{last_error}"
                , id="3"
            )
            )

        return state

    def _llm_node_final_answer_generator(self,state:SqlAgentState)-> SqlAgentState:

        final_answer_generator_prompt_template = PromptTemplate(
            input_variables=['query_execution_result'],
            template=self.system_prompt_dict['final_answer_generator']
        )
        final_answer_generator_prompt = final_answer_generator_prompt_template.format(
            query_execution_result=state['messages'][-1].content
        )

        system_msg = SystemMessage(content=final_answer_generator_prompt, id="2")
        state['messages'].append(system_msg)

        human_msg = HumanMessage(state['question'], id="1")

        response = self.llm.invoke([system_msg, human_msg])
        sql_agent_response = response.content

        state["messages"].append(AIMessage(content=sql_agent_response, id="3"))
        state["response"] = sql_agent_response

        return state

    def get_sql_agent(self)->CompiledStateGraph:
        builder = StateGraph(SqlAgentState)

        builder.add_node('SQL query generator',self._llm_node_sql_query_generator)
        builder.add_node('Final answer generator', self._llm_node_final_answer_generator)

        builder.add_edge(START, "SQL query generator")
        builder.add_edge("SQL query generator",'Final answer generator')
        builder.add_edge('Final answer generator',END)

        return builder.compile()
