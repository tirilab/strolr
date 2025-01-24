#!/usr/bin/env python
# coding: utf-8

# In[153]:


import streamlit as st
import streamlit.components.v1 as components
from langchain.vectorstores.pgvector import PGVector
from pgvector.psycopg2 import register_vector
from langchain.llms import OpenAI
from openai import OpenAI
import openai
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
import time

# layout
st.set_page_config(layout="wide")
left_column = st.sidebar
right_column = st

# Display the name of the app

#col1, col2, col3 = st.columns(3)
#right_column.image('logo.png', use_column_width=False)
left_co, cent_co,last_co = st.columns(3)
logo = "logo.png"
with cent_co:
    st.image('logo.png')

right_column.write(r"$\textsf{\Large An LLM-Enabled Chatbot To Support Pregnant Women’s Information Seeking From Trustworthy Sources}$")

# Sidebar - Resources
left_column.link_button("Resources", "https://mh-n.github.io/strolr/")
# Add more buttons as needed ("how to use" button)

# Main content area - Chat
right_column.title('Chat with Strolr')
right_column.write('Please enter your name and click on the button below to start the chat.')

user_input = right_column.text_input("Name")
openai_api_key = "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"

os.environ["OPENAI_API_KEY"] = openai_api_key

COLLECTION_NAME = "strolr_test"
CONNECTION_STRING = PGVector.connection_string_from_db_params(
     driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
     host=os.environ.get("PGVECTOR_HOST", "localhost"),
     port=int(os.environ.get("PGVECTOR_PORT", "5432")),
     database=os.environ.get("PGVECTOR_DATABASE", "vectordb"),
     user=os.environ.get("PGVECTOR_USER", "user"),
     password=os.environ.get("PGVECTOR_PASSWORD", "temp"),
)

#def generate_response(input_text):
#    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
#    return llm(input_text)


# In[22]:


import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="vectordb",
    user="user",
    password="temp")


# In[35]:


#cur = conn.cursor()
#cur.execute("""
#cur.execute("SELECT COUNT(*) as cnt FROM langchain_pg_embedding;")
#"""
#)


# In[127]:


#num_records = cur.fetchone()[0]
#print("Number of vector records in table: ", num_records,"\n")


# In[128]:


#cur.execute("SELECT * FROM langchain_pg_embedding LIMIT 1;")
#records = cur.fetchall()
#print("First record in table: ", records)


# In[123]:


embeddings = OpenAIEmbeddings()
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)


# In[125]:


#query = "Is it safe for my unborn baby if I eat raw fish during pregnancy?"
#similar = store.similarity_search_with_score(query, k=3)


# In[133]:


def summarize_doc(query, resources):

    messages = [{"role": "system",
                 "content": "You are acting as a friendly clinician who is speaking to a patient. Do not say you are an AI."}]

    prompt = """Answer the user's prompt or question:

    {query}

    by summarizing the following 3 texts:
    
    {resource1}
    
    {resource2}
    
    {resource3}

    Keep your answer direct and concise. Use only the information in the provided text to answer the query.""".format(query=query, resource1=resources[0][0].page_content, resource2=resources[1][0].page_content, resource3=resources[2][0].page_content)

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    #return response.choices[0]["message"]["content"]
    return response


# In[139]:


#from openai import OpenAI
#client = OpenAI()
#response = summarize_doc(query, similar)


# In[142]:


#response.choices[0].message.content


# In[140]:


def generate_response(user_query):
    similar = store.similarity_search_with_score(user_query, k=3)
    client = OpenAI()
    response = summarize_doc(user_query, similar)
    return response.choices[0].message.content


# In[141]:


### PAST FXN

#if right_column.button('Start Chat'):
#    right_column.write(f'Hi {user_input}! I am Strolr. I am here to help you with your queries related to pregnancy.')
#    right_column.write('Please enter your query in the text box below and click on the button below to get the answer.')

#    user_query = right_column.text_input("Query")
#    if right_column.button('Get Answer'):
        #if not openai_api_key.startswith('sk-'):
        #    right_column.warning('Please enter your OpenAI API key!', icon='⚠')
        #else:
#        right_column.write("Hi")
        #response = generate_response(user_query)
        #right_column.write('Answer:')
        #right_column.write(response)


# In[158]:


#What should my target blood sugar be during pregnancy?
#similar = store.similarity_search_with_score("What are the typical blood sugar levels during pregnancy?", k=3)


# In[151]:


@st.cache_resource
def load_chain():

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()

    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0)

    # Load our local FAISS index as a retriever
    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

    retriever = store.as_retriever(search_kwargs={"k": 3})

    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")

    # Create system prompt
    template = """
        You are acting as a friendly clinician who is speaking to a patient.
        Do not say you are an AI.
        The patient is looking for information related to pregnancy. 
        This patient has below a proficient health literacy level based on the National Assessment of Adult Literacy. Please adjust your response accordingly.
        This patient reads at a 6th grae reading level. Please adjust your response accordingly.
        Only provide the answer to questions you can find answers to in the database. If the information is not in the database, just apologize and say that you do not know the answer.
        Don't try to make up an answer.
        If the question is not related to pregnancy or childcare, politely inform them that you are tuned to only answer questions about pregnancy and childcare.
        When you respond to the patient, please cite the source of your answer by referring to the original embeddings data source.
       
        {context}
        Question: {question}
        Helpful Answer:"""

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                  retriever=retriever,
                                                  memory=memory,
                                                  get_chat_history=lambda h : h,
                                                  verbose=True)

    # Add systemp prompt to chain
    # Can only add it at the end for ConversationalRetrievalChain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain


# In[154]:



#if right_column.button('Start Chat'):
#    right_column.write(f'Hi {user_input}! I am Strolr. I am here to help you with your queries related to pregnancy.')
#    right_column.write('Please enter your query in the text box below and click on the button below to get the answer.')
chain = load_chain()

if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": f'Hi {user_input}! I am Strolr. I am here to help you with your queries related to pregnancy.'}]

# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
# Chat logic
if query := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = chain({"question": query})
        response = result['answer']
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


# In[ ]:




