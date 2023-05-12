import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
openaiapikey = st.secrets["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings()
logger = logging.getLogger(__name__)

def create_db(docs, embeddings=embeddings):
    if not docs:
        logger.error("No documents to create the database.")
        return None
    try:
        database  = FAISS.from_documents(docs, embeddings).as_retriever()
    except Exception as e:
        logger.error(f"An error occurred while creating the database: {e}")
        return None
    return database






