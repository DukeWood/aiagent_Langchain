import os
from sys import modules
from dotenv import load_dotenv

# load .env file
load_dotenv()


# Retrieve LangSmith API key from environment variables
langsmith_api_key = os.getenv("langsmith_api_key")

# Check if the API key is retrieved correctly
if not langsmith_api_key:
    raise ValueError("LangSmith API key is not set in the environment variables.")


# get the bot token from the .env file
openai_api_key = os.getenv("openai_api_key")
serpapi_api_key = os.getenv("serpapi_api_key")

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {question}\nThought:{agent_scratchpad}")

# # import langchain hub
# from langchain import hub

# # initialize the hub
# # hub = HuggingFaceHub(repo_id="google/flan-t5-base",
                     
# # from hub to get ReAct prompt
# prompt = hub.pull("hwchase17/react")

# print (prompt)
                     
from langchain_openai import OpenAI

llm = OpenAI(openai_api_key=openai_api_key)

from langchain_community.utilities import SerpAPIWrapper

from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor

# Initialize SerpAPI with the key
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="please search content while LLMs don't have the related content"
    ),
]

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# agent_executor.invoke({"input":"what is microbit?"})
question = input("What is your question?" )
agent_executor.invoke({"question": question})

