from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

def process_node(state:AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI - {response.content}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

user_input = ""
while user_input != "exit":
    user_input = input("Enter: ")
    app.invoke({"messages" : [HumanMessage(content=user_input)]})
    print("\n")