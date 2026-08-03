from langchain_community.vectorstores import FAISS
from google import genai
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

import os
from dotenv import load_dotenv
load_dotenv()  
API_key = os.environ["GOOGLE_API_KEY"]
client = genai.Client(api_key=API_key)

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key= API_key
)

from pathlib import Path

vectordb_file_path = "faiss_index"
def create_vector_db():
    # Load data from FAQ sheet
    #loader = CSVLoader(file_path='q_a_db.csv', source_column="prompt")

    BASE_DIR = Path(__file__).parent

    csv_path = BASE_DIR / "q_a_db.csv"

    loader = CSVLoader(
        file_path=str(csv_path),
        source_column="prompt"
    )
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(
        vectordb_file_path,
        instructor_embeddings,
        allow_dangerous_deserialization=True
    )


    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't have enough informations regarding this matter. In case, I will report it to the Support team to call you for further details." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        api_key=API_key,
        temperature=0.7
    )
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("How can I access to the support team"))
