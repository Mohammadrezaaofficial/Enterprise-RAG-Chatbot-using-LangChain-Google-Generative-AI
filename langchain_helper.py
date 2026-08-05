import os
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI)
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)
# for model in client.models.list():
#     print(model.name)

vectordb_file_path = "faiss_index"

##################################################
# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key=API_KEY
)


# Create Vector Database
def create_vector_db():

    loader = CSVLoader(
        file_path="q_a_db.csv",
        source_column="prompt"
    )

    documents = loader.load()

    vectordb = FAISS.from_documents(
        documents,
        embeddings
    )

    vectordb.save_local(
        vectordb_file_path
    )



# QA Chain
def get_qa_chain():

    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
        }
    )

    prompt = PromptTemplate(

        input_variables=[
            "context",
            "question"
        ],

        template=""" You are a professional customer support assistant. Use ONLY the provided context. Never invent information. If the answer is not available inside the context, reply EXACTLY with:
"I don't have enough information regarding this matter. I will report it to our support team, who will contact you using the information you provided."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    api_key=API_KEY,
    temperature=0.2
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )

    return qa


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain.invoke({
        "query":"How can I contact support?"
    }))
