import streamlit as st
st.title('Strolr')
st.write('Welcome to Strolr, An LLM-enabled Chat Web-app To Support Pregnant Women’s Information Seeking From Trustworthy Sources.')
st.write('Please enter your name and click on the button below to start the chat.')

from langchain.llms import OpenAI
openai_api_key = st.sidebar.text_input('Strolr')
def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  st.info(llm(input_text))
  with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
user_input = st.text_input("Name")
if st.button('Start Chat'):
    st.write('Hi ' + user_input + '! I am Strolr. I am here to help you with your queries related to pregnancy.')
    st.write('Please enter your query in the text box below and click on the button below to get the answer.')
    user_input1 = st.text_input("Query")
    if st.button('Get Answer'):
        st.write('Answer:')