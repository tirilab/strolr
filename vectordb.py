import getpass
import os
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
import pickle
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector


def retrieve_sources(query):

    # CONNECT TO RDS
    connection = "postgresql+psycopg://langchain:langchain@strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com:5432/postgres"
    collection_name = "strolr_docs"
    embeddings = OpenAIEmbeddings()
    
    store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,)

    sources = store.similarity_search_with_score(query, k=3)
    print(sources)

    return sources 


retrieve_sources('What can I do to protect my baby?')