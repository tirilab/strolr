#!/usr/bin/env python
# coding: utf-8

import re
from langchain import hub
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from openai import OpenAI
from langchain_core.documents import Document
import openai
import numpy as np
from langchain_openai import OpenAIEmbeddings
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from IPython.display import Markdown, display
from streamlit_extras.switch_page_button import switch_page
from datetime import date
from langchain_openai import ChatOpenAI
import time
import psycopg2
from langchain_core.messages import AIMessage
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# layout
st.set_page_config(layout="wide")
left_column = st.sidebar
right_column = st

# Display the name of the app
st.markdown(
    """
    <style>
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .responsive-text {
        font-size: 1.8em;  /* Increase font size */
        text-align: center;
        margin-top: 10px;   /* Add spacing below logo */
        word-wrap: break-word;
        white-space: normal;
    }
    @media (max-width: 768px) {
        .responsive-text {
            font-size: 1.4em;  /* Adjust font size for smaller screens */
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Columns for layout
left_co, cent_co, last_co = st.columns([1, 2, 1])  # Center column wider to allow space for text
small_logo = "strolr_bot.svg"
# Centered logo and text in the same column with alignment
logo = "LOGO_FINAL.png"
with cent_co:
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.image(logo)
    st.markdown(
        '<div class="responsive-text">An LLM-Enabled Chatbot To Support Pregnant Women’s Information Seeking From Trustworthy Sources</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# Sidebar
with st.sidebar.expander("How to use"):
    st.write("""
        **How to use Strolr**\n\n

        Type your question in the chat and press *Enter* to start the conversation.\n
        Strolr is only designed to answer the questions related to healthy pregnancy.\n
        To view the source of information provided by Strolr, click the link in the response.\n
        To browse all sources, click the *Resources* button below.\n

        Your chat history will be lost after you close the session.
        Click *Download chat history* button, if you wish to save the conversaion.\n

        Strolr is **NOT** designed to give you medical advice.\n
        **If you are experiencing an emergency, call 9-1-1 immediately.**
    """)

left_column.link_button("Resources", "https://tirilab.github.io/strolr/", help = "Click to browse resources")

# Main content area - Chat
right_column.title('Chat with Strolr')
right_column.write('Please enter your name to start the chat.')
today = date.today()

user_input = right_column.text_input("Name")
openai_api_key = os.environ["OPENAI_API_KEY"]


# Create a string for downloadable CHAT HISTORY
if user_input != '':
    chat_hist_download = user_input + '\'s chat history on ' + str(today) + '\n'
    username_hist = user_input
else:
    chat_hist_download = 'Your chat history on ' + str(today) + '\n'
    username_hist = 'You'
    
messages_for_download = []
chat_hist = ''



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# FORMATTING RESPONSES
def format_response(responses):
    source_documents = responses["context"]
    source_content = [doc.page_content for doc in source_documents]
    source_metadata = [doc.metadata for doc in source_documents]

    used_urls = []

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

# CONNECT TO RDS

COLLECTION_NAME = "strolr_docs"
CONNECTION_STRING = PGVector.connection_string_from_db_params(
     driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
     host=os.environ.get("PGVECTOR_HOST", "strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com"),
     port=int(os.environ.get("PGVECTOR_PORT", "5432")),
     database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
     user=os.environ.get("PGVECTOR_USER", "langchain"),
     password=os.environ.get("PGVECTOR_PASSWORD", "langchain"),
)

conn = psycopg2.connect(
    host="strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com",
    database="postgres",
    user="langchain",
    password="langchain")
connection = "postgresql+psycopg://langchain:langchain@strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com:5432/postgres"
collection_name = "strolr_docs"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)
    

@st.cache_resource
#CHAIN
def load_chain_with_sources():

    retriever = store.as_retriever(search_type="similarity_score_threshold", search_kwargs = {"k":3, "score_threshold":0.35})


    llm = ChatOpenAI(temperature = 0.5, model = "gpt-4o-mini")

    # Create system prompt
    template = """
        You are acting as a friendly clinician who is speaking to a patient.
        Do not say you are an AI. Don't say you're a clinician ever to the user.
        The patient is looking for information related to healthy pregnancy according to {input}. Assuming the user is pregnant, if the question is not about pregnancy health, you can apologize, say you don't>        This patient has below a proficient health literacy level based on the National Assessment of Adult Literacy. Please adjust your response accordingly.
        This patient reads at a 6th grade reading level. Please adjust your response accordingly.
        Only provide the answer to {input} you can find answers to in the {context}. Do your best to comprehensively summarize all the relevant information in the {context}. If the information is not there, j>        Never provide resources if they are not relevant to the user's question.
        Never give a response in any language besides the English language even if the user requests it.
        If the answer is not in the {context}, say that you don't know in a kind way or give them a suggestion on a different question to ask.
        Do your best to understand typos, casing, and framing of questions.
        {input}
        {context}
       """
    # Create the Conversational Chain
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        
    ]
)
    rag_chain_from_docs = (
        {
            "input": lambda x: x["input"],  # input query
            "context": lambda x: format_docs(x["context"]),  # context
        }
        | prompt  # format query and context into prompt
        | llm  # generate response
        | StrOutputParser()  # coerce to string
    )

    # Pass input query to retriever
    retrieve_docs = (lambda x: x["input"]) | retriever

    # Below, we chain `.assign` calls. This takes a dict and successively
    # adds keys-- "context" and "answer"-- where the value for each key
    # is determined by a Runnable. The Runnable operates on all existing
    # keys in the dict.
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )

    # Invoke the RAG chain with the question
    return chain 

# USER CHATS
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
                                          f'Hi {user_input}! 👋 I\'m Strolr, your go-to chatbot for all things related to healthy pregnancies! While I\'m here to help answer your questions, it\'s important to>
                                          '\n'
                                          'My mission is to provide quick and reliable information by tapping into a database filled with trustworthy pregnancy sources. I\'m your virtual pregnancy encyclopedi>
                                          '\n'
                                          'Feel free to ask me about topics like nutrition, fetal development, prental care and more that you can find in the Resources tab to the left. If you have a pressing >
                                          '\n'
                                          'Remember, I\'m here to assist and inform, but your healthcare provider should be your primary source for personalized advice. Let\'s embark on this journey together,>
                                          '\n'
                                          '*NOTE:* your chat history will **not** be saved when you close the session. If you wish to save your conversation, click **Download chat history** button in the chat>

        
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

            # CHATBOT RESPONSE
            with st.chat_message("assistant", avatar=small_logo):
                message_placeholder = st.empty()
                message_placeholder.markdown('...')
                # Send user's question to our chain
                context = "\n".join([message["content"] for message in st.session_state.messages])
                formatted_query = {'input': query}
                result = chain.invoke(formatted_query)
                #metadata = [msg for msg in result]
                print(result)
                #response = metadata[0][1]
                response = format_response(result)
                match_str = r"(don''t know)|(do not know)|(cannot answer)|(can''t answer)|(I am sorry, I don''t know)|(I''m sorry, I don''t know)|(I''m sorry, I can''t answer)"
                if re.match(match_str, response):
                    response = re.sub(r"(Sources used:.*)", '', response, flags=re.DOTALL)
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



st.cache_data.clear() 

