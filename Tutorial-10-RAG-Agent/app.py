from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

pdf_path = "EML_Vision_v1.0.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"\nPDF loaded successfully: {pdf_path}")
except Exception as e:
    print(f"\n Error loading pdf: {str(e)}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
page_chunks = text_splitter.split_documents(pages)
collection_name = "eml_vision"

try:
    vector_store = Chroma.from_documents(
        documents=page_chunks,
        embedding=embeddings,
        collection_name=collection_name
    )
    print(f"ChromaDB vector store created.")

except Exception as e:
    print(f"Error occurred while setting up chromadb vector database: {str(e)}")
    raise

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5}
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from the ELM Vision v1.0 document."""
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the ELM Vision v1.0 document."
    
    # Storing the retrieved document's page content as a list.
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}\n{doc.page_content}")
    
    #Using .join() to combine the list of strings from results into a single string.
    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state["messages"][-1]
    return isinstance(result, AIMessage) and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI Assistant who answers questions about the EML Vision v1.0 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the vision of EML. You can make multiple calls if needed.
If you need to look up for some information before asking a follow up question, you are allowed to do that.
Please always cite the specific parts of the document you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

#LLM Agent
def call_llm(state:AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    conversation_history = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + conversation_history
    response = llm.invoke(messages)
    return {"messages": [response]}

#Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided.')}")

        if not t["name"] in tools_dict:
            print(f"\nTool: {t['name']} does not exist")
            result = "Incorrect Tool Name. Please Retry and Select tool from the list of Available tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Tool Execution Completed. Back to the model!")
    return {'messages': results}
    
graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever", take_action)
graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True:"retriever",
        False: END
    }
)
graph.add_edge("retriever", "llm")

app = graph.compile()

def run_agent():
    print("\n ===== RAG Agent =====")
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "stop", "quit"]:
            break
        messages = [HumanMessage(content=user_input)]
        result = app.invoke({"messages": messages})
        print("\n ===== Answer =====")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    run_agent()