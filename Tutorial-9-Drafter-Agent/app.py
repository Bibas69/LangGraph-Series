# =========================
# Imports
# =========================

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
import os


# =========================
# Environment setup
# =========================

# Load variables from .env into os.environ
load_dotenv()

# (Optional, but harmless if already set)
# ChatGroq reads GROQ_API_KEY from environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# =========================
# Global document storage
# =========================

# This variable holds the current document text
document_content = ""


# =========================
# Graph State Definition
# =========================

class AgentState(TypedDict):
    """
    The graph state.
    - messages is a list of chat messages
    - add_messages tells LangGraph to APPEND new messages automatically
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =========================
# Tools
# =========================

@tool
def update_content(content: str) -> str:
    """
    Update the document with the provided full content.
    """
    global document_content
    document_content = content
    return f"Document updated successfully.\nCurrent content:\n{document_content}"


@tool
def post_content(filename: str) -> str:
    """
    Save the current document to a .txt file and finish.
    """
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(document_content)

        return f"Document saved successfully as '{filename}'."

    except Exception as e:
        return f"Error saving document: {str(e)}"


# Tool list
tools = [update_content, post_content]


# =========================
# Model
# =========================

model = ChatGroq(
    model="openai/gpt-oss-20b"
).bind_tools(tools)


# =========================
# Agent Node
# =========================

def agent(state: AgentState) -> dict:
    """
    This node:
    - Builds the system prompt
    - Collects user input (first run vs later runs)
    - Calls the model
    - Returns ONLY NEW messages
    """

    # System instructions for the LLM
    system_prompt = SystemMessage(
        content=f"""
You are a Drafter, a helpful writing assistant.

Rules:
- If the user wants to update or modify the document, use the tool `update_content`
- If the user wants to save and finish, use the tool `post_content`
- Always show the current document after modifications

Current document content:
{document_content if document_content else "(empty document)"}
"""
    )

    # FIRST RUN: no messages yet
    if not state["messages"]:
        user_message = HumanMessage(
            content="Hey. I want you to help me with a document."
        )
    else:
        # FOLLOW-UP RUNS: ask user for input in terminal
        user_input = input("\nWhat would you like to do with the document? ")
        user_message = HumanMessage(content=user_input)

    # Full message list sent to the model
    all_messages = (
        [system_prompt]
        + list(state["messages"])
        + [user_message]
    )

    # Call the LLM
    response = model.invoke(all_messages)

    # Print AI response to console
    print(f"\nAI: {response.content}")

    # IMPORTANT:
    # Return ONLY new messages
    # LangGraph will append them automatically
    return {
        "messages": [user_message, response]
    }


# =========================
# Continue / Stop Logic
# =========================

def should_continue(state: AgentState) -> str:
    """
    Decide whether to:
    - continue looping
    - or END the graph
    """

    # Look at messages from newest to oldest
    for message in reversed(state["messages"]):

        # If a tool message says document was saved → END
        if isinstance(message, ToolMessage):
            content_lower = message.content.lower()
            if "saved" in content_lower and "document" in content_lower:
                return "end"

    # Otherwise, keep going
    return "continue"


# =========================
# Pretty printing tool output
# =========================

def print_messages(messages):
    """
    Print tool outputs nicely.
    """
    for message in messages:
        if isinstance(message, ToolMessage):
            print(f"\n[TOOL RESULT]\n{message.content}")


# =========================
# Build the Graph
# =========================

graph = StateGraph(AgentState)

# Nodes
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

# Edges
graph.add_edge(START, "agent")

# Agent → Tools (only if tool_calls exist)
graph.add_conditional_edges(
    "agent",
    lambda state: "tools" if getattr(state["messages"][-1], "tool_calls", None) else "continue",
    {
        "tools": "tools",
        "continue": "agent",
    },
)

# Tools → either loop or end
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

# Compile the graph
app = graph.compile()


# =========================
# Runner
# =========================

def run_document_agent():
    print("\n===== DRAFTER STARTED =====")

    # Initial empty state
    state = {"messages": []}

    # Stream execution
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n===== DRAFTER FINISHED =====")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    run_document_agent()
    