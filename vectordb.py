import os
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain.document_loaders import PyPDFLoader
import psycopg2
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import SeleniumURLLoader
import pandas as pd
import numpy as np
import openai
import pandas as pd
import numpy as np
import openai
import nltk
nltk.download('punkt')

with open('docs.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# RDS Connection details
connection = "postgresql+psycopg://langchain:langchain@strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com:5432/postgres"
collection_name = "strolr_docs"

vector_store = PGVector.from_documents(
    embeddings=embeddings,
    documents = data,
    collection_name=collection_name,
    connection_string=connection,
    use_jsonb=True,
)
