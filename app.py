import os
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

CHROMA_PATH = "chroma"
st.set_page_config(page_title="RAG Query System")
                   
def response(question):
    loader = CSVLoader("oscar_text.csv", encoding="utf-8")
    documents = loader.load()
    db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    template = """You are a helpful AI assistant and your goal is to answer questions as accurately as possible based on the context provided. Be concise and just include the response:

    context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain.invoke(question)

def response_without_rag(question):
    template = """ Answer the question below:

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = (
          prompt
        | model
        | StrOutputParser()
    )
    return chain.invoke(question)

st.subheader("Question:")
question=st.text_input(label="",key="question")
response_chain = response(question)
response_wout_rag = response_without_rag(question)
submit=st.button("Submit")
if submit:
    st.subheader("The Response using RAG is")
    st.write(response_chain)
    st.subheader("The Response without using RAG is")
    st.write(response_wout_rag)

