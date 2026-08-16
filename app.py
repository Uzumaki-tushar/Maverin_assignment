import os
import sqlite3
import uuid
import streamlit as st
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq

# Environment Variables setup
# To run locally securely or deploy to Streamlit Cloud, it's best to use Streamlit Secrets
import streamlit as st

# Pull keys from Streamlit secrets (if available) or fallback to environment variables
try:
    os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", os.environ.get("LANGCHAIN_API_KEY", ""))
    os.environ["QDRANT_API_KEY"] = st.secrets.get("QDRANT_API_KEY", os.environ.get("QDRANT_API_KEY", ""))
    os.environ["QDRANT_URL"] = st.secrets.get("QDRANT_URL", os.environ.get("QDRANT_URL", ""))
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "3gpp-rag-chatbot"
except Exception:
    pass # Fallback to local environment variables if st.secrets fails

# Streamlit Page Config
st.set_page_config(page_title="3GPP RAG Chatbot", page_icon="🤖", layout="centered")

# Initialize backend (Cached to avoid re-initializing Qdrant and LangGraph repeatedly)
@st.cache_resource
def init_backend():
    # 1. Embeddings & Vector Store
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    qdrant_url = os.environ.get("QDRANT_URL", "")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")

    # Prioritize the committed local qdrant_db directory if present
    if os.path.exists("qdrant_db"):
        try:
            vectorstore = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name="3gpp_standards",
                path="qdrant_db"
            )
        except Exception:
            # Fallback if qdrant_db is locked by another process on serverless/Streamlit Cloud
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp()
            target_path = os.path.join(temp_dir, "qdrant_db")
            shutil.copytree("qdrant_db", target_path, ignore=shutil.ignore_patterns('.lock', '*.lock'))
            vectorstore = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name="3gpp_standards",
                path=target_path
            )
    elif qdrant_url and qdrant_api_key:
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name="3gpp_standards",
            url=qdrant_url,
            api_key=qdrant_api_key
        )
    else:
        raise RuntimeError("No qdrant_db folder found and no QDRANT_URL/QDRANT_API_KEY provided.")
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. LLM Setup with Fallbacks
    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "qwen/qwen3.6-27b"
    ]
    llms = [ChatGroq(temperature=0, model=m) for m in models]
    llm = llms[0].with_fallbacks(fallbacks=llms[1:])

    # 3. LangGraph Setup
    class GraphState(TypedDict):
        messages: Annotated[list, add_messages]
        context: List[str]
        hallucination_check: str

    def retrieve(state: GraphState):
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        return {"context": docs}

    class GradeDocuments(BaseModel):
        binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    def grade_documents(state: GraphState):
        question = state["messages"][-1].content
        docs = state["context"]
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        system = "You are a grader assessing relevance of a retrieved document to a user question. If the document contains keyword(s) or semantic meaning related to the question, grade it as 'yes'."
        grade_prompt = PromptTemplate(template=system + "\n\nRetrieved document: \n\n {document} \n\n User question: {question}", input_variables=["document", "question"])
        retrieval_grader = grade_prompt | structured_llm_grader
        
        relevant_docs = []
        for d in docs:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            if score.binary_score == "yes":
                relevant_docs.append(d)
        
        if not relevant_docs:
            return {"messages": [AIMessage(content="The provided 3GPP specifications do not contain information to answer this query.")]}
        return {"context": relevant_docs}

    def generate(state: GraphState):
        question = state["messages"][-1].content
        docs = state["context"]
        context_str = "\n\n".join(f"[{d.metadata.get('source', 'unknown')} - {d.metadata.get('section', 'unknown')}]: {d.page_content}" for d in docs)
        
        system = """You are a strict 3GPP standards expert. Answer ONLY using the provided context. 
        If the context does not contain the answer, reply exactly with 'The provided 3GPP specifications do not contain information to answer this query.' 
        Always cite the source document and section for every claim. Do not use outside knowledge."""
        prompt = f"{system}\n\nContext:\n{context_str}\n\nQuestion: {question}"
        response = llm.invoke(prompt)
        return {"messages": [response]}

    class GradeHallucinations(BaseModel):
        binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

    def check_hallucination(state: GraphState):
        docs = state["context"]
        generation = state["messages"][-1].content
        
        structured_llm_grader = llm.with_structured_output(GradeHallucinations)
        system = "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. Give a binary score 'yes' or 'no'."
        hallucination_prompt = PromptTemplate(template=system + "\n\nSet of facts: \n\n {documents} \n\n LLM generation: {generation}", input_variables=["documents", "generation"])
        hallucination_grader = hallucination_prompt | structured_llm_grader
        
        docs_str = "\n\n".join([d.page_content for d in docs])
        score = hallucination_grader.invoke({"documents": docs_str, "generation": generation})
        
        if score.binary_score == "yes":
            return {"hallucination_check": "pass"}
        else:
            return {"hallucination_check": "fail", "messages": [AIMessage(content="The generated answer could not be verified against the 3GPP context and was blocked to prevent hallucination.")]}

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    def decide_to_generate(state):
        if state["messages"][-1].content == "The provided 3GPP specifications do not contain information to answer this query.":
            return "end"
        return "generate"

    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"generate": "generate", "end": END})
    workflow.add_edge("generate", "check_hallucination")

    def handle_hallucination(state):
        return END

    workflow.add_conditional_edges("check_hallucination", handle_hallucination)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

# Initialize backend
app = init_backend()

# Initialize Streamlit Session State for multi-threaded chat history
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if st.session_state.thread_id not in st.session_state.sessions:
    st.session_state.sessions[st.session_state.thread_id] = []

# Sidebar for Chat History
with st.sidebar:
    st.header("💬 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.thread_id = new_id
        st.session_state.sessions[new_id] = []
        st.rerun()
        
    st.divider()
    st.subheader("Previous Sessions")
    
    for thread in list(st.session_state.sessions.keys()):
        name = f"Session {thread[:8]}"
        if st.button(name, key=f"btn_{thread}", use_container_width=True):
            st.session_state.thread_id = thread
            st.rerun()

# --- Streamlit UI ---
st.title("📡 3GPP RAG Chatbot")
st.markdown("A high-precision, near-zero hallucination chatbot for Telecom 3GPP standards.")

# Display chat history for current active thread
active_messages = st.session_state.sessions.get(st.session_state.thread_id, [])
for msg in active_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a question about 3GPP standards..."):
    # Store and render user message
    st.session_state.sessions[st.session_state.thread_id].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process via LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Searching 3GPP specifications and analyzing..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            final_response = ""
            # Stream output
            for output in app.stream(inputs, config=config, stream_mode="values"):
                message = output["messages"][-1]
                if isinstance(message, AIMessage):
                    final_response = message.content
            
            st.markdown(final_response)
            st.session_state.sessions[st.session_state.thread_id].append({"role": "assistant", "content": final_response})
