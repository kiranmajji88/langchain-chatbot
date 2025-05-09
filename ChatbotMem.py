from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit
st.title("Conversational Chatbot with Memory")

# Session-based memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize LLM and chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

chain = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=True
)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Ask a question:")

if user_input:
    response = chain.predict(input=user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display conversation
for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")

# Debug: Show memory contents
with st.expander("Debug: Memory Variables"):
    st.json(st.session_state.memory.load_memory_variables({}))
