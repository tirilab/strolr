# import streamlit as st
# from langchain.llms import OpenAI

# # layout
# st.set_page_config(layout="wide")
# left_column = st.sidebar
# right_column = st

# # Sidebar - Resources
# left_column.title('Resources')
# left_column.write('Resource 1: Description')
# left_column.write('Resource 2: Description')
# # Add more resources as needed

# # Main content area - Chat
# right_column.title('Chat with Strolr')
# right_column.write('Welcome to Strolr, an LLM-enabled Chat Web-app To Support Pregnant Women’s Information Seeking From Trustworthy Sources.')
# right_column.write('Please enter your name and click on the button below to start the chat.')

# user_input = right_column.text_input("Name")
# openai_api_key = left_column.text_input('OpenAI API Key')

# def generate_response(input_text):
#     llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
#     return llm(input_text)

# if right_column.button('Start Chat'):
#     right_column.write(f'Hi {user_input}! I am Strolr. I am here to help you with your queries related to pregnancy.')
#     right_column.write('Please enter your query in the text box below and click on the button below to get the answer.')

#     user_query = right_column.text_input("Query")
#     if right_column.button('Get Answer'):
#         if not openai_api_key.startswith('sk-'):
#             right_column.warning('Please enter your OpenAI API key!', icon='⚠')
#         else:
#             response = generate_response(user_query)
#             right_column.write('Answer:')
#             right_column.info(response)

import streamlit as st
from langchain.llms import OpenAI
import base64
import time

# layout
st.set_page_config(layout="wide")
left_column = st.sidebar
right_column = st



# Information Popup
if left_column.button('ℹ️'):
    st.toast('This app is designed to assist pregnant women with their queries related to pregnancy.  To use the chat, enter your name, start the chat, and ask your questions in the text box provided. Click on the "Get Answer" button to receive a response from Strolr.!')
    time.sleep(.5)


# info_popup = left_column.button('ℹ️')
# if info_popup:
#     st.sidebar.text(info_text)
#     #create a button to close the popup when clicked
#     close_popup = left_column.button('Close')
#     if close_popup:
#         st.sidebar.text('')
        

# Sidebar - Resources
left_column.link_button("Resources", "https://streamlit.io/gallery")
# Add more resources as needed
   
# Main content area - Chat
right_column.title('Chat with Strolr')
right_column.write('Welcome to Strolr, an LLM-enabled Chat Web-app To Support Pregnant Women’s Information Seeking From Trustworthy Sources.')
right_column.write('Please enter your name and click on the button below to start the chat.')

user_input = right_column.text_input("Name")
# openai_api_key = left_column.text_input('OpenAI API Key')

chat_history = []  # For storing chat history

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    return llm(input_text)

if right_column.button('Start Chat'):
    right_column.write(f'Hi {user_input}! I am Strolr. I am here to help you with your queries related to pregnancy.')
    right_column.write('Please enter your question in the text box below and click on the button below to get the answer.')

    user_query = right_column.text_input("Question")
    if right_column.button('Get Answer'):
        if not openai_api_key.startswith('sk-'):
            right_column.warning('Please enter your OpenAI API key!', icon='⚠')
        else:
            response = generate_response(user_query)
            right_column.write('Answer:')
            right_column.info(response)
            chat_history.append((user_query, response))

# def get_binary_file_downloader_html(file_path, file_label='File'):
#     with open(file_path, 'rb') as file:
#         contents = file.read()
#     encoded_file = base64.b64encode(contents).decode('utf-8')
#     href = f'<a href="data:file/txt;base64,{encoded_file}" download="{file_path}">{file_label}</a>'
#     return href

# # Download Chat History
# if st.button("Download Chat History"):
#     # Logic to download chat history as a text file (Simple placeholder)
#     with open('chat_history.txt', 'w') as file:
#         for query, response in chat_history:
#             file.write(f'Query: {query}\nResponse: {response}\n\n')
#     st.markdown(get_binary_file_downloader_html('chat_history.txt', 'Chat History'), unsafe_allow_html=True)

    

# Download Chat History
if st.button("Download Chat History"):
    # Logic to download chat history as a text file
    
    st.download_button(label="Download Chat History", data=chat_history, file_name="chat_history.txt", mime="text/plain")

# # # Download chat history
# st.download_button('Download some text', chat_history)

# download_history = right_column.button('Download Chat History')
# if download_history:
#     with open('chat_history.txt', 'w') as file:
#         for query, response in chat_history:
#             file.write(f'Query: {query}\nResponse: {response}\n\n')
#     right_column.write('Chat history downloaded successfully!')
