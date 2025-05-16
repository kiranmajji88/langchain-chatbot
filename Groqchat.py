import streamlit as st  # UI framework
import os  # Access environment variables
import time  # Time performance tracking
from dotenv import load_dotenv  # Load .env variables

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===================== Functions =====================

def initialize_vectorstore():
    """Initializes the embedding and vector database only once."""
    if "vectors" not in st.session_state:
        # Load and process docs
        embeddings = OllamaEmbeddings()
        loader = WebBaseLoader("https://docs.smith.langchain.com/")
        raw_docs = loader.load()

        # Text chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(raw_docs[:50])  # Limit for performance

        # FAISS index
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Cache in session
        st.session_state.vectors = vectorstore


def build_retrieval_chain():
    """Builds the LangChain retrieval-augmented chain."""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Provide the most accurate response:
        <context>
        {context}
        </context>
        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    return create_retrieval_chain(retriever, document_chain)


def run_chatbot(chain):
    """Handles the user prompt and displays result."""
    user_input = st.text_input("Ask something from LangChain docs:")

    if user_input:
        start = time.process_time()
        response = chain.invoke({"input": user_input})
        duration = time.process_time() - start

        st.markdown(f"**Response Time:** {duration:.2f}s")
        st.write(response["answer"])

        with st.expander("Retrieved Document Chunks"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.markdown("---")

# ===================== Streamlit UI =====================

st.title("🔍 Chat with LangChain Docs using Groq LLM")
initialize_vectorstore()
rag_chain = build_retrieval_chain()
run_chatbot(rag_chain)
