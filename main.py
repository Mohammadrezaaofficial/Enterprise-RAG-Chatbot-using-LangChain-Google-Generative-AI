import os
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Hello, How can I help you? 🌱")

# Create the vector database automatically if it doesn't exist
if not os.path.exists("faiss_index"):
    with st.spinner("Creating knowledge base... Please wait."):
        create_vector_db()
        print(os.getcwd())
        print(os.listdir("."))

question = st.text_input("Question:")

if question:
    chain = get_qa_chain()
    response = chain.invoke({"query": question})  # Recommended for newer LangChain

    st.header("Answer")
    st.write(response["result"])
