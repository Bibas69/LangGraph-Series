from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

conversation_history = []
user_input = input("Enter: ")
while(user_input != "exit"):
    conversation_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages":conversation_history})
    print("\n")
    conversation_history = result["messages"]
    user_input = input("Enter: ")
