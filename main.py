import os
import csv
import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Customer Support Chatbot")


# Create FAISS database if it doesn't exist
if not os.path.exists("faiss_index"):
    with st.spinner("Creating knowledge base..."):
        create_vector_db()


# Session State
if "customer" not in st.session_state:
    st.session_state.customer = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Customer Information Form
if st.session_state.customer is None:

    st.subheader("Customer Information")

    with st.form("customer_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        start = st.form_submit_button("Start Chat")

    if start:

        if name.strip() and (email.strip() or phone.strip()):
            st.session_state.customer = {
                "name": name,
                "email": email,
                "phone": phone
            }

            st.rerun()

        else:
            st.warning("Please enter your name and either email or phone.")

    st.stop()


# Load QA chain once
@st.cache_resource
def load_chain():
    return get_qa_chain()

chain = load_chain()


# Show Chat History
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat Input
with st.form("chat_form", clear_on_submit=True):

    question = st.text_input(
        "Ask your question..."
    )

    send = st.form_submit_button("Send")


# Generate Answer
if send and question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    history = ""

    for msg in st.session_state.messages:
        history += f"{msg['role']}: {msg['content']}\n"

    with st.spinner("Thinking..."):

        response = chain.invoke({
            "query":
            f"""
Conversation History:

{history}

Current Question:

{question}
"""
        })

    answer = response["result"]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Save chat log
    file_exists = os.path.exists("chat_logs.csv")

    with open(
        "chat_logs.csv",
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Name",
                "Email",
                "Phone",
                "Question",
                "Answer"
            ])

        writer.writerow([
            st.session_state.customer["name"],
            st.session_state.customer["email"],
            st.session_state.customer["phone"],
            question,
            answer
        ])

    st.rerun()