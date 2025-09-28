import os
import getpass

from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain_community.utilities.requests import RequestsWrapper
from langchain import hub


from tools.knowledge_base import KnowledgeBaseTool
from tools.prayer_timer import PrayerTimerTool
from tools.tourist_recommendation import TouristRecommendationTool
from tools.trip_budget_planner import TripBudgetPlannerTool
from llms.gemini import Gemini


def agent_init():
    if 'GOOGLE_API_KEY' not in os.environ:
        api_key = getpass.getpass('Enter your Google API key: ')
        os.environ['GOOGLE_API_KEY'] = api_key

    
    gemini = Gemini(api_key=api_key, temperature=0.7)
    llm = gemini.llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        return_messages=True
    )
    
    data_tool = RequestsWrapper()
    get_tool = RequestsGetTool(
        requests_wrapper=data_tool,
        allow_dangerous_requests=True
    )

    prayer_tool = PrayerTimerTool(llm=llm, get_tool=get_tool)
    knowledge_tool = KnowledgeBaseTool(get_tool=get_tool)
    tourist_tool = TouristRecommendationTool(llm=llm)
    budget_tool = TripBudgetPlannerTool(llm=llm)

    tools = [
        prayer_tool,
        knowledge_tool,
        tourist_tool,
        budget_tool
    ]

    prompt = hub.pull("hwchase17/react-chat")

    agent_chain = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent = AgentExecutor(
        agent=agent_chain,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True
    )

    return agent
