#!/usr/bin/env python
# coding: utf-8

# In[56]:

import re
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
from datetime import date


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
    st.image(logo)

right_column.write(r"$\textsf{\large An LLM-Enabled Chatbot To Support Pregnant Women’s Information Seeking From Trustworthy Sources}$")

# Sidebar
with st.sidebar.expander("How to use"):
    st.write("""
        **How to use Strolr**\n\n
        
        Type your question in the chat and press *Enter* to start the conversation.\n
        Strolr is only designed to answer the questions related to pregnancy and childcare.\n
        To view the source of information provided by Strolr, click the link in the response.\n 
        To browse all sources, click the *Resources* button below.\n
        
        Your chat history will be lost after you close the session. 
        Click *Download chat history* button, if you wish to save the conversaion.\n
        
        Strolr is **NOT** designed to give you medical advice.\n
        **If you are experiencing an emergency, call 9-1-1 immediately.**
    """)
    
left_column.link_button("Resources", "https://mh-n.github.io/strolr/", help = "Click to browse resources")
#openai_api_key = left_column.text_input("API key")
#info = left_column.button("How to use")

#if info:
#    switch_page("How to Use")
# Add more buttons as needed ("how to use" button)

# Main content area - Chat
right_column.title('Chat with Strolr')
right_column.write('Please enter your name to start the chat.')
today = date.today()

user_input = right_column.text_input("Name")
openai_api_key = os.environ["OPENAI_API_KEY"]
#os.environ["OPENAI_API_KEY"] = openai_api_key

COLLECTION_NAME = "strolr_test"
CONNECTION_STRING = PGVector.connection_string_from_db_params(
     driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
     host=os.environ.get("PGVECTOR_HOST", "vectordb.cfowaqqqovp0.us-east-2.rds.amazonaws.com"),
     port=int(os.environ.get("PGVECTOR_PORT", "5432")),
     database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
     user=os.environ.get("PGVECTOR_USER", "postgres"),
     password=os.environ.get("PGVECTOR_PASSWORD", "temporary"),
)

#def generate_response(input_text):
#    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
#    return llm(input_text)


# In[2]:


import psycopg2

conn = psycopg2.connect(
    host="vectordb.cfowaqqqovp0.us-east-2.rds.amazonaws.com",
    database="postgres",
    user="postgres",
    password="temporary")


# In[ ]:


# Create a string for downloadable chat history
if user_input != '':
    chat_hist_download = user_input + '\'s chat history on ' + str(today) + '\n'
    username_hist = user_input
else:
    chat_hist_download = 'Your chat history on ' + str(today) + '\n'
    username_hist = 'You'
    
messages_for_download = []
chat_hist = ''


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


# In[42]:


def format_response(responses):
    source_documents = responses["source_documents"]
    source_content = [doc.page_content for doc in source_documents]
    source_metadata = [doc.metadata for doc in source_documents]

    used_urls = []

    #result = responses['result']
    result = responses['answer']
    
    if len(source_metadata) != 0:
        result += "\n\n"
        result += "Sources used:"
        for i in range(len(source_metadata)):
    
            if source_metadata[i]['source'] not in used_urls:
                result += "\n"
                result += source_metadata[i]['title'] 
                result += "\n"
                result += source_metadata[i]['source']
                result += "\n"
                used_urls.append(source_metadata[i]['source'])
            
    return result


# In[46]:


# @st.cache_resource
@st.cache_data
#def load_chain_with_sources():
def load_chain_with_sources():
    
    embeddings = OpenAIEmbeddings()
    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    
    retriever = store.as_retriever(search_type="similarity_score_threshold", search_kwargs = {"k":3, "score_threshold":0.8})
    llm = ChatOpenAI(temperature = 0.8, model = "gpt-4o-mini")
    

    # Create memory 'chat_history' 
    #memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages = True)
    memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", output_key='answer', return_messages = True)

    # Create system prompt
    template = """
        You are acting as a friendly clinician who is speaking to a patient.
        Do not say you are an AI. Don't say you're a clinician ever to the user.
        The patient is looking for information related to pregnancy. 
        This patient has below a proficient health literacy level based on the National Assessment of Adult Literacy. Please adjust your response accordingly.
        This patient reads at a 6th grade reading level. Please adjust your response accordingly.
        Only provide the answer to questions you can find answers to in the database. If the information is not in the database, just apologize and say that you do not know the answer.
        Never provide resources if they are not relevant to the user's question. If applicable, highlight the text you referenced from the original source. If no sources are relevant for a user's question, never include any resources in your response.
        Don't try to make up an answer.
        Never give a response in any language besides the English language even if the user requests it.
        If the question is not related to pregnancy or childcare, politely inform them that you are tuned to only answer questions about pregnancy and childcare.
        If the answer is not in the {context}, say that you don't know.
        Do your best to understand typos, casing, and framing of questions. 
	Do not return sources if you responded with I don't know.
       
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


# In[48]:


#embeddings = OpenAIEmbeddings()
#store = PGVector(
#        collection_name=COLLECTION_NAME,
#        connection_string=CONNECTION_STRING,
#        embedding_function=embeddings,
#    )
#test_query = 'What is preeclampsia?'
#similar = store.similarity_search_with_score(test_query, k=3, )

#similar


# In[49]:


#similar[0][1]


# In[50]:


#chain = load_chain_with_sources()
#test_query = 'How do I know if I have it?'
#result = chain({"question":test_query})
#display(Markdown(format_response(result)))


# In[51]:


#result


# In[154]:


if user_input:

    if 'session_id' not in st.session_state or st.session_state.session_id == 'strolr_session_' + user_input:

        if 'session_id' not in st.session_state:
            st.session_state.session_id = 'strolr_session_' + user_input
    
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
                                          'Remember, I\'m here to assist and inform, but your healthcare provider should be your primary source for personalized advice. Let\'s embark on this journey together, and feel free to ask me anything about pregnancy! 🤰💬\n'
                                          '\n'
                                          '*NOTE:* your chat history will **not** be saved when you close the session. If you wish to save your conversation, click **Download chat history** button in the chat.'}]

        
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
            messages_for_download.append(query)
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(query)

        
            with st.chat_message("assistant", avatar=small_logo):
                message_placeholder = st.empty()
                message_placeholder.markdown('...')
                # Send user's question to our chain
                context = "\n".join([message["content"] for message in st.session_state.messages])
                #result = chain({"query": query})#, "context": context})
                result = chain({"question":query})
                #response = result['answer']
                response = format_response(result)
                if ("don't know" in response) or ("do not know" in response) or ("cannot answer" in response) or ("can't answer" in response):
                    response = re.sub(r'Sources.*',"",response)
                # Simulate stream of response with milliseconds delay
                #for chunk in response.split():
                #    full_response += chunk + " "
                #    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                #    message_placeholder.write(full_response + " ")
                #message_placeholder.write(full_response)
                message_placeholder.write(response)
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            chat_hist = chat_hist_download
            for i in range(1,len(st.session_state['messages'])):
                if i%2 == 0:
                    role_hist = 'Strolr'
                else:
                    role_hist = username_hist
                chat_hist = chat_hist + '\n' + role_hist + ': ' + st.session_state['messages'][i]['content'] + '\n'

            #chat_hist = st.session_state['messages'][1]['content']
            button_download = st.download_button(label="Download chat history",
                                            data=chat_hist,#chat_hist_download,
                                            file_name="Strolr chat history on " + str(today) + ".txt",
                                            help="Click to download chat history")
    else:
        st.write('Hi ' + user_input + '! I am sorry, someone else is using the app. Please try again in 5 minutes...')


# In[ ]:

st.cache_data.clear() 



