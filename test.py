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

connection = "postgresql+psycopg://langchain:langchain@strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com:5432/postgres"
collection_name = "strolr_docs"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
#query = "What should I do if I have a post-partum depression?"
query = "Is it safe for my unborn baby if I eat raw fish during pregnancy?"
similar = vector_store.similarity_search_with_score(query, k=3)
vector = embeddings.embed_query(query)

for doc in similar:
    print(doc)
    print('\n')

similar[0][0]