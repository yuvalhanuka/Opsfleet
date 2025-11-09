from dotenv import load_dotenv
from datetime import datetime
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, filter_messages
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from helper_functions import *
from plot_agent import PlotAgent, PlotAgentState
from sql_agent import SqlAgent, SqlAgentState
from .state import DataAnalysisAgentState, SupervisorOutput, ExplorerOutput


class DataAnalysisAgent:
    def __init__(self):
        self.script_directory = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        self.system_prompt_dict = self._get_system_prompt_dict()
        self.llm_name = self._get_config()
        self.llm = self._get_llm(model_name = self.llm_name)
        self.sql_agent = SqlAgent().get_sql_agent()
        self.plot_agent = PlotAgent().get_plot_agent()


    def _get_system_prompt_dict(self):
        system_prompt_dict_path = os.path.join(self.script_directory, 'files','system_prompts.json')
        with open(system_prompt_dict_path, 'rb') as f:
            system_prompt_dict = json.load(f)
        return system_prompt_dict

    def _get_config(self):
        config_path = os.path.join(self.script_directory, 'files','config.json')
        with open(config_path, 'rb') as f:
            config = json.load(f)
        return config['llm_name']

    @staticmethod
    def _get_llm(model_name:str):
        load_dotenv()
        return ChatGoogleGenerativeAI(model=model_name)

    def _llm_node_supervisor(self,state: DataAnalysisAgentState)->DataAnalysisAgentState:

        supervisor_system_prompt_template = PromptTemplate(
            input_variables=['chat_history','format_instructions','sql_description'],
            template=self.system_prompt_dict['supervisor']
        )

        memory = filter_messages(state['messages'], include_ids=['0', '1'])
        memory = memory[-10:]

        pydantic_parser = PydanticOutputParser(pydantic_object=SupervisorOutput)
        format_instructions = pydantic_parser.get_format_instructions()

        sql_description = get_tables_information()

        supervisor_system_prompt = supervisor_system_prompt_template.format(
            chat_history=memory,
            format_instructions=format_instructions,
            sql_description = sql_description
        )

        system_msg = SystemMessage(content=supervisor_system_prompt, id="2")


        user_question = state["user_question"]
        human_msg = HumanMessage(content=f"user_question:\n{user_question}", id="0")

        state['messages'].append(human_msg)
        state['messages'].append(system_msg)

        response = self.llm.invoke([system_msg, human_msg])

        supervisor_output = pydantic_parser.parse(response.content)
        state['supervisor_decision'] = supervisor_output

        if state['supervisor_decision'].decision == 'response':

            chat_response = state['supervisor_decision'].response
            state['messages'].append(AIMessage(content= chat_response, id = '1'))
            state['chat_response'] = chat_response

        else:

            explore = state['supervisor_decision'].explore
            state['messages'].append(AIMessage(content=explore, id='3'))

        return state

    @staticmethod
    def _router_node_supervisor_decision(state: DataAnalysisAgentState)->str:
        return state['supervisor_decision'].decision

    def _llm_node_explorer(self,state: DataAnalysisAgentState)->DataAnalysisAgentState:

        explorer_system_prompt_template = PromptTemplate(
            input_variables=['current_time','format_instructions','sql_description'],
            template=self.system_prompt_dict['explorer']
        )
        pydantic_parser = PydanticOutputParser(pydantic_object=ExplorerOutput)

        format_instructions = pydantic_parser.get_format_instructions()
        sql_description = get_tables_information()

        explorer_system_prompt = explorer_system_prompt_template.format(
            format_instructions=format_instructions,
            sql_description = sql_description,
            current_time = datetime.now().strftime("%Y-%d-%m %H:%M")
        )

        system_msg = SystemMessage(content=explorer_system_prompt, id="2")
        state['messages'].append(system_msg)

        exploration_instruction = state['supervisor_decision'].explore
        human_msg = HumanMessage(content=exploration_instruction)

        response = self.llm.invoke([system_msg, human_msg])
        explorer_output = pydantic_parser.parse(response.content)

        state['explorer_decision'] = explorer_output
        state['messages'].append(AIMessage(content=explorer_output.model_dump_json(indent=2),id='3'))

        return state

    def _sql_agent_node(self,state: DataAnalysisAgentState)->DataAnalysisAgentState:

        sql_agent_result = []
        for question in state['explorer_decision'].questions_for_sql_agent:
            sql_agent_state: SqlAgentState = {
                "question": question,
                "messages": []
                }
            response = self.sql_agent.invoke(sql_agent_state)
            sql_agent_result.append({question:response['response']})

        return {'sql_agent_response':sql_agent_result}

    def _plot_agent_node(self,state: DataAnalysisAgentState)->DataAnalysisAgentState:

        if (state['explorer_decision'].plot_description == '') or (state['explorer_decision'].plot_description is None):
            return {'plot_agent_response': None, 'plot_file_path': None}

        else:
            question = state['supervisor_decision'].explore
            plot_description = state['explorer_decision'].plot_description
            plot_agent_state: PlotAgentState = {
                "question": question,
                "messages": [],
                "plot_description": plot_description
            }
            res = self.plot_agent.invoke(plot_agent_state)
            return {'plot_agent_response': res['plot_analysis'], 'plot_file_path': res['saved_plot_path']}

    def _llm_node_final_answer_generator(self, state: DataAnalysisAgentState) -> DataAnalysisAgentState:
        supervisor_system_prompt_template = PromptTemplate(
            input_variables=['questions_and_answers_fetched_from_sql', 'plot_analysis_fetched_from_plot_agent'],
            template=self.system_prompt_dict['final_answer_generator']
        )

        sql_blocks = []
        for qa in state.get('sql_agent_response', []) or []:
            for q, a in qa.items():
                sql_blocks.append(f"Q: {q}\nA: {a}")
        questions_and_answers_fetched_from_sql = "\n\n".join(sql_blocks) if sql_blocks else "None"

        plot_analysis_fetched_from_plot_agent = state.get('plot_agent_response') or "None"

        final_answer_system_prompt = supervisor_system_prompt_template.format(
            questions_and_answers_fetched_from_sql=questions_and_answers_fetched_from_sql,
            plot_analysis_fetched_from_plot_agent=plot_analysis_fetched_from_plot_agent
        )

        system_msg = SystemMessage(content=final_answer_system_prompt, id="2")
        state['messages'].append(system_msg)

        human_msg = HumanMessage(content=state['supervisor_decision'].explore)

        response = self.llm.invoke([system_msg, human_msg])
        final_answer = response.content

        state['messages'].append(AIMessage(content=final_answer, id='1'))
        state['chat_response'] = final_answer

        return state

    def get_data_analysis_agent(self)->CompiledStateGraph:

        builder = StateGraph(DataAnalysisAgentState)
        builder.add_node('supervisor',self._llm_node_supervisor)
        builder.add_node('explorer',self._llm_node_explorer)
        builder.add_node('SQL agent',self._sql_agent_node)
        builder.add_node('Plot agent', self._plot_agent_node)
        builder.add_node('Final answer generator',self._llm_node_final_answer_generator)

        builder.add_edge(START,'supervisor')
        builder.add_conditional_edges(
            source="supervisor",
            path=self._router_node_supervisor_decision,
            path_map={
                    'explore':'explorer',
                    'response':END
            }
        )
        builder.add_edge( 'explorer','SQL agent')
        builder.add_edge('explorer', 'Plot agent')

        builder.add_edge( 'SQL agent','Final answer generator')
        builder.add_edge( 'Plot agent','Final answer generator')

        builder.add_edge( 'Final answer generator',END)
        return builder.compile()
