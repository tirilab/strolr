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



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# RDS Connection details
connection = "postgresql+psycopg://langchain:langchain@strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com:5432/strolrdb"
collection_name = "strolr_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

with open('docs.pkl', 'rb') as f:
    data = pickle.load(f)

vector_store.add_documents(data)