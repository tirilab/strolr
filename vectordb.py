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
import os
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import PyPDFLoader
import psycopg2
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import SeleniumURLLoader
import pandas as pd
import numpy as np
import openai
import nltk
nltk.download('punkt')

url_file = pd.read_csv("url_resources.csv", sep = ',', encoding = 'cp1252', header = 0, names = ['Resource_group', 'Title', 'URL', 'd'])[['Resource_group', 'Title', 'URL']]
url_file = url_file.dropna()
urls = np.array(url_file['URL'])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# RDS Connection details
connection = "postgresql+psycopg://langchain:langchain@strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com:5432/postgres"
collection_name = "strolr_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

ct = 1
all_documents = []

for url in urls:
    url_array = [url]
    print("\nLoading raw document ", ct, "...")
    loader = SeleniumURLLoader(urls=url_array)
    raw_documents = loader.load()
    
    print("Splitting text...")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    
    documents = text_splitter.split_documents(raw_documents)
    all_documents.extend(documents)
    
    ct += 1

import getpass
import os
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

vector_store.add_documents(all_documents)
query = "Is it safe for my unborn baby if I eat raw fish during pregnancy?"
similar = vector_store.similarity_search_with_score(query, k=3)
print(similar)