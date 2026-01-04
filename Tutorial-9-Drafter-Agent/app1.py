from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

document_content = []  #Global variable to store the document content.

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_content(content: str) -> str:
    """Update the document with provided full content."""
    global document_content
    document_content = content
    return f"Document updated successfully. Current content:\n {document_content}"

@tool
def save_content(filename: str) -> str:
    """Save the current document to a .txt and finish."""
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    
    try:
        with open(filename, "w") as file:
            file.write(document_content)
        return f"Document saved successfully as {filename}."
    
    except Exception as e:
        return f"Some error occurred. Error: {str(e)}"
    
tools = [update_content, save_content]

llm = ChatGroq(model="openai/gpt-oss-20b").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_message = SystemMessage(f"""
        You are a drafter, a helpful writing assistant.
        Rules:
        - If the user wants to update or modify the document, use the tool `update_content`
        - If the user wants to save and finish, use the tool `post_content`
        - Always show the current document after modifications
        
        Current Document Content: {document_content if document_content else ('empty document')}
    """)

    if not state["messages"]:
        ai_message = "Greet the user and ask them what they want help with."
        user_message = AIMessage(content=ai_message)

    else:
        user_input = input("AI - Tell me what you want to do with the document:")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_message] + list(state["messages"]) + [user_message]
    response = llm.invoke(all_messages)

    print(f"\nAI- {response.content}")
    return {
        "messages": [user_message, response]
    }

def should_continue(state: AgentState) -> str:
    messages = state["messages"]

    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            content_lower = message.content.lower()
            if "save" in content_lower and "document" in content_lower:
                return "exit"
            
    return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools=tools))

graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "exit": END
    }
)

app = graph.compile()

def print_message(messages):
    """Function to print he message in a readable format."""
    if not messages:
        return
    for message in messages:
        if isinstance(message, ToolMessage):
            print(f"\nTool Result: {message.content}")

def run_document_agent():
    print("\n##### Drafter Started #####")

    state = {"message": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])

    print("##### Drafter Finished #####")

if __name__ == "__main__":
    run_document_agent()