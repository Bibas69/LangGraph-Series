from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """This is an add function that adds two numbers together."""
    return a+b

@tool
def subtract(a:int, b:int):
    """This is a subtract function that subtracts two numbers."""
    return a-b

@tool
def multiply(a:int, b:int):
    """This is a multiply function that multliplies two numbers."""
    return a*b

def divide(a:int, b:int):
    """This is a divide function that divides two numbers."""
    return a/b

tools = [add, subtract, multiply, divide]

model = ChatGroq(model="openai/gpt-oss-20b").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are a helpful AI Assistant. Please answer my query to the best of your performance.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node("agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple()):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 3 + 4. Subtract 20 - 10. Multiply 2 and 3. Divide 10 by 2.")]}
print_stream(app.stream(inputs, stream_mode="values"))