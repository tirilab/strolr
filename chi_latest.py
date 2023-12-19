import streamlit as st
from langchain.llms import OpenAI

# Set up the layout
st.set_page_config(layout="wide")
left_column = st.sidebar
right_column = st

# Sidebar - Resources
left_column.title('Resources')
left_column.write('Resource 1: Description')
left_column.write('Resource 2: Description')
# Add more resources as needed

# Main content area - Chat
right_column.title('Chat with Strolr')
right_column.write('Welcome to Strolr, an LLM-enabled Chat Web-app To Support Pregnant Women’s Information Seeking From Trustworthy Sources.')
right_column.write('Please enter your name and click on the button below to start the chat.')

user_input = right_column.text_input("Name")
openai_api_key = left_column.text_input('OpenAI API Key')

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    return llm(input_text)

if right_column.button('Start Chat'):
    right_column.write(f'Hi {user_input}! I am Strolr. I am here to help you with your queries related to pregnancy.')
    right_column.write('Please enter your query in the text box below and click on the button below to get the answer.')

    user_query = right_column.text_input("Query")
    if right_column.button('Get Answer'):
        if not openai_api_key.startswith('sk-'):
            right_column.warning('Please enter your OpenAI API key!', icon='⚠')
        else:
            response = generate_response(user_query)
            right_column.write('Answer:')
            right_column.info(response)
