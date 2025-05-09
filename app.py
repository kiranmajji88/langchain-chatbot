from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv


#Below is how to trace the monitoring results

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Now create a prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are my assistant. Please answer user queries"),
        ("user", "Question:{question}")
    ]

)

# Call the streamlit framework

st.title("Langchain Demo with OPENAI API")
input_text = st.text_input("Search the topic you want")

# call the OpenAI LLM

llm=ChatOpenAI(model="gpt-3.5-turbo")
output_Parser = StrOutputParser()
chain=prompt|llm|output_Parser

if input_text:
    st.write(chain.invoke({"question":input_text}))




