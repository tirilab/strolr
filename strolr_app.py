#!/usr/bin/env python
# coding: utf-8

# In[55]:


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
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.chains import RetrievalQA
from IPython.display import Markdown, display
from streamlit_extras.switch_page_button import switch_page

import time

# layout
st.set_page_config(layout="wide")
left_column = st.sidebar
right_column = st

# Display the name of the app

left_co, cent_co,last_co = st.columns(3)
logo = "LOGO_FINAL.png"
small_logo = "strolr_bot.svg"
with cent_co:
    st.image('logo.png')

right_column.write(r"$\textsf{\Large An LLM-Enabled Chatbot To Support Pregnant Women’s Information Seeking From Trustworthy Sources}$")

# Sidebar - Resources
left_column.link_button("Resources", "https://mh-n.github.io/strolr/")
openai_api_key = left_column.text_input("API key")
#info = left_column.button("How to use")

#if info:
#    switch_page("How to Use")
# Add more buttons as needed ("how to use" button)

# Main content area - Chat
right_column.title('Chat with Strolr')
right_column.write('Please enter your name to start the chat.')

user_input = right_column.text_input("Name")
#openai_api_key = ### INSERT OPEN API KEY HERE

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


# In[2]:


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


# In[3]:


#embeddings = OpenAIEmbeddings()
#store = PGVector(
#    collection_name=COLLECTION_NAME,
#    connection_string=CONNECTION_STRING,
#    embedding_function=embeddings,
#)


# In[4]:


#query = "Is it safe for my unborn baby if I eat raw fish during pregnancy?"
#similar = store.similarity_search_with_score(query, k=3)


# In[6]:


#retriever = store.as_retriever(search_kwards = {"k":3})
#llm = ChatOpenAI(temperature = 0.0, model = "gpt-3.5-turbo-16k")


# In[7]:


#qa_stuff = RetrievalQA.from_chain_type(
#    llm=llm,
#    chain_type="stuff",
#    retriever=retriever,
#    return_source_documents = True,
#    verbose=True
#)


# In[20]:


#responses = qa_stuff({"query": query})
#source_documents = responses["source_documents"]
#source_content = [doc.page_content for doc in source_documents]
#source_metadata = [doc.metadata for doc in source_documents]

#used_urls = []

# Construct a single string with the LLM output and the source titles and urls
#def construct_result_with_sources():
#    result = responses['result']
#    result += "\n\n"
#    result += "Sources used:"
#    for i in range(len(source_metadata)):
    
#        if source_metadata[i]['source'] not in used_urls:
#            result += "\n\n"
#            result += source_metadata[i]['title']
#            result += "\n\n"
#            result += source_metadata[i]['source']
#            used_urls.append(source_metadata[i]['source'])
            
#    return result

#display(Markdown(construct_result_with_sources()))


# In[48]:


### TEST CELL
#test_query = "What should my blood sugar be during pregnancy"
#res = qa_stuff({"query": test_query})
#res


# In[64]:


def format_response(responses):
    source_documents = responses["source_documents"]
    source_content = [doc.page_content for doc in source_documents]
    source_metadata = [doc.metadata for doc in source_documents]

    used_urls = []

    #result = responses['result']
    result = responses['answer']
    result += "\n\n"
    result += "Sources used:"
    for i in range(len(source_metadata)):
    
        if source_metadata[i]['source'] not in used_urls:
            result += "\n\n"
            result += source_metadata[i]['title']
            result += "\n\n"
            result += source_metadata[i]['source']
            used_urls.append(source_metadata[i]['source'])
            
    return result


# In[57]:


@st.cache_resource
#def load_chain_with_sources():
def load_chain_with_sources():
    
    embeddings = OpenAIEmbeddings()
    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    
    retriever = store.as_retriever(search_kwards = {"k":3})
    llm = ChatOpenAI(temperature = 0.0, model = "gpt-3.5-turbo-16k")
    

    # Create memory 'chat_history' 
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages = True)

    # Create system prompt
    template = """
        You are acting as a friendly clinician who is speaking to a patient.
        Do not say you are an AI.
        The patient is looking for information related to pregnancy. 
        This patient has below a proficient health literacy level based on the National Assessment of Adult Literacy. Please adjust your response accordingly.
        This patient reads at a 6th grade reading level. Please adjust your response accordingly.
        Only provide the answer to questions you can find answers to in the database. If the information is not in the database, just apologize and say that you do not know the answer.
        Don't try to make up an answer.
        If the question is not related to pregnancy or childcare, politely inform them that you are tuned to only answer questions about pregnancy and childcare.
       
        {context}
        Question: {question}
        Helpful Answer:"""

    # Create the Conversational Chain
    #chain = RetrievalQA.from_chain_type(
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory = memory,
        get_chat_history=lambda h :h,
        return_source_documents = True,
        verbose=False
    )
    
    # Add systemp prompt to chain
    # Can only add it at the end for ConversationalRetrievalChain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    
    
    #chain.combine_documents_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
    
    #print(chain)
    
    return chain


# In[72]:


#chain = load_chain_with_sources()
#test_query = "When did World War 2 end?"
#result = chain({"question":test_query})
#format_response(result)


# In[73]:


#result


# In[154]:


if user_input:
    
    chain = load_chain_with_sources()

    if 'messages' not in st.session_state:
        # Start with first message from assistant
        st.session_state['messages'] = [{"role": "assistant", 
                                      "content": '🌟 **Welcome to Strolr - Your Pregnancy Info Companion!** 🌟 \n'
                                      '\n'
                                      f'Hi {user_input}! 👋 I\'m Strolr, your go-to chatbot for all things pregnancy-related! While I\'m here to help answer your questions, it\'s important to note that I\'m not a substitute for professional medical advice. Always consult with your healthcare provider for personalized guidance.\n'
                                      '\n'
                                      'My mission is to provide quick and reliable information by tapping into a database filled with trustworthy pregnancy sources. I\'m your virtual pregnancy encyclopedia, designed to make finding information a breeze.\n'
                                      '\n'
                                      'Feel free to ask me about topics like nutrition, prenatal care, common symptoms, and much more. If you have a pressing question, I\'m here to help point you in the right direction based on reliable sources.\n'
                                      '\n'
                                      'Remember, I\'m here to assist and inform, but your healthcare provider should be your primary source for personalized advice. Let\'s embark on this journey together, and feel free to ask me anything about pregnancy! 🤰💬'}]

        
    # Display chat messages from history on app rerun
    # Custom avatar for the assistant, default avatar for user
    for message in st.session_state.messages:
        if message["role"] == 'assistant':
            with st.chat_message(message["role"], avatar=small_logo):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat logic
    if query := st.chat_input("Your question about pregnancy"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        
        with st.chat_message("assistant", avatar=small_logo):
            message_placeholder = st.empty()
            # Send user's question to our chain
            context = "\n".join([message["content"] for message in st.session_state.messages])
            #result = chain({"query": query})#, "context": context})
            result = chain({"question":query})
            #response = result['answer']
            response = format_response(result)
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




