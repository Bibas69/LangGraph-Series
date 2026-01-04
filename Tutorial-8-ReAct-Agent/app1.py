from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """This is a function that adds two numbers together."""
    return a+b

tools = [add]

model = ChatGroq(model="openai/gpt-oss-20b").bind_tools(tools)

def model_call(state:AgentState):
    system_prompt = SystemMessage(content="You are a helpful AI Assistant. Please answer my query to the best of your performance.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "exit"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("model", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model")
graph.add_conditional_edges(
    "model",
    should_continue,
    {
        "continue": "tools",
        "exit": END
    }
)
graph.add_edge("tools", "model")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user", "What tools do you have access to?")]}
print_stream(app.stream(inputs, stream_mode="values"))