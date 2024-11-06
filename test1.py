import openai
from openai import OpenAI
import streamlit as st
from streamlit_chat import message
import langchain
import pandas as pd
from io import StringIO
import time
import numpy as np
import os 
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

def load_chain_with_sources():

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    connection = "postgresql+psycopg://langchain:langchain@strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com:5432/postgres"
    collection_name = "strolr_docs"
    vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,)

    retriever = vector_store.as_retriever(search_kwards = {"k":3})
    llm = ChatOpenAI(temperature = 0.7, model = "gpt-4o-mini")
    

    # Create memory 'chat_history' 
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages = True)

    # Create system prompt
    template = """
        You are acting as a friendly clinician who is speaking to a patient.
        Do not say you are an AI or a clinician.
        The patient is looking for information related to health conditions. 
        This patient has below a proficient health literacy level based on the National Assessment of Adult Literacy. Please adjust your response accordingly.
        This patient reads at a 6th grade reading level. Please adjust your response accordingly.
        Only provide the answer to questions you can find answers to in the database. If the information is not in the database, just apologize and say that you do not know the answer.
        Don't try to make up an answer.
        If the question is not related to health conditions, politely inform them that you are tuned to only answer questions about health conditions.
       
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



#Initial page setup
st.set_page_config(
    page_title="KIT", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# mh-n.github.io" 
    }
)
right = st
st.markdown(
    """
    <style>
    /* Center the chat input at the bottom */
    div.streamlit-chat-input textarea {
        display: flex;
        flex-direction: column;
        width: 90%;
        margin: 0 0 10px 10px;
        z-index: 1;
    }

    /* Align chat buttons next to chat input */
    div.streamlit-chat-buttons {
        position: fixed;
        bottom: 0;
        right: 10px;
        z-index: 1;
    }

    /* Style the family history checkboxes */
    div.streamlit-checkbox label {
        font-size: 1rem;
        color: #4a4a4a;
        margin: 5px;
    }

    /* Other customization like adding background color to chat */
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 15px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
### SIDEBAR
with st.sidebar:
    st.header("_KIT_, :blue[ the family health history chatbot] :wave::robot_face::speech_balloon:")
    with st.expander("Read KIT introduction"):
        st.info("Hi, I'm KIT! My job is to collect information about your family health history, but all your responses will remain anonymous and confidential. \n\n I will ask you a few quick questions. This module asks you about your family’s medical history. You will be shown a list of conditions and asked to select those that certain family members have or had. If you need help understanding the conditions, you can always ask me! \n\n Understanding your family’s experiences with medical issues can tell us a lot about what kinds of medical issues might be related to your genetics. Genetics has to do with traits that are passed down from generation to generation in a family. You will be asked questions about your family. Think only of people you are related to by blood including those living or deceased.  \n\n If you have have a previous conversation, feel free to upload it using the file uploader below.")
    with st.expander("KIT functions"):
        st.subheader("Conversing with KIT", divider= "blue")
        st.markdown("Please use the buttons to select relevant question options or type your answers in the text input field. Click submit to move forward in the conversation.")
        st.subheader("Text functions",  divider="blue")
        st.markdown("Type or press buttons for the following commands:\n\n :red[BACK ]: Return to the previous question \n\n :red[HELP ]: Get a list of commands for what KIT can help you with.")
        st.markdown("At any time in the conversation, you may use the text input to ask KIT a clarifying question about conditions or what details you should provide about your family history if you're ever unsure. \n\n")
        st.info("_i.e. What does anemia mean?_")
        st.markdown("KIT was designed to answer questions about health and family history, retrieving information from a database of consumer health sources and definitions. KIT will not be able to respond to questions outside this scope.")
    with st.expander("Family history file uploader"):
        st.write("Upload a .txt file of your family history conversation using the file uploader below! \n\nThis will help KIT prompt you with questions for family history you have already noted.")

        uploaded_file =  st.file_uploader(label = 'Upload', label_visibility="hidden")
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            st.write(bytes_data)

            # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.write(stringio)

            # To read file as string:
            string_data = stringio.read()
            st.write(string_data)

            # Can be used wherever a "file-like" object is accepted:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)
###


### KIT 
# Initialize the necessary session variables\
if 'FHx_upload' not in st.session_state:
    st.session_state.FHx_upload = []
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'kit_history' not in st.session_state:
    st.session_state.kit_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # This records chat history
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0 # This records chat history
if 'options' not in st.session_state:
    st.session_state.options = []
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = 0
if 'current_condition_category' not in st.session_state:
    st.session_state['current_condition_category'] = None
if 'family_members' not in st.session_state:
    st.session_state['family_members'] = []
if 'specific_conditions' not in st.session_state:
    st.session_state.specific_conditions =  []
if 'condition_index' not in st.session_state:
    st.session_state.condition_index = 0
if 'family_member_index' not in st.session_state:
    st.session_state['family_member_index'] = 0
if 'current_question' not in st.session_state:
    st.session_state.current_question = ''
if 'condition_index' not in st.session_state:
    st.session_state['condition_index'] = 0

def stream_data(text, delay):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)
def conversation_started():
    st.session_state.conversation_started = 1

# On greeting
if 'greetings' not in st.session_state:  
    st.session_state.greetings = False
if st.session_state.greetings == False:  # Greet user only if greeting hasn't been shown
    kit = st.chat_message('assistant', avatar='🤖')
    intro = "Hi, I'm KIT! My job is to collect information about your family health history, but all your responses will remain anonymous and confidential."
    intro2 = "I will ask you a few quick questions. This module asks you about your family’s medical history. You will be shown a list of conditions and asked to select those that certain family members have or had. If you need help understanding the conditions, you can always ask me by typing your question into the chat input bar!"
    intro3 = "Understanding your family’s experiences with medical issues can tell us a lot about what kinds of medical issues might be related to your genetics. Genetics has to do with traits that are passed down from generation to generation in a family. You will be asked questions about your family. Think only of people you are related to by blood including those living or deceased."
    intro4 = "If you have a previous conversation, feel free to upload it using the file uploader located in the sidebar. Additionally, if you'd ever like to go back at any time or get help, please type those in the chat input or press the buttons on the side of the chat input bar."

    # Stream each intro message
    kit.write_stream(stream_data(intro, 0.1))
    kit.write_stream(stream_data(intro2, 0.1))
    kit.write_stream(stream_data(intro3, 0.1))
    kit.write_stream(stream_data(intro4, 0.1))

    kit.write('Click the _Start_ button to start our family health history chat!')
    
    # Store intro in session state to prevent retyping
    st.session_state.chat_history.append({"role": "assistant", "content": intro})
    st.session_state.chat_history.append({"role": "assistant", "content": intro2})
    st.session_state.chat_history.append({"role": "assistant", "content": intro3})
    st.session_state.chat_history.append({"role": "assistant", "content": intro4})

    st.session_state.kit_history.append({"role": "assistant", "content": intro})
    st.session_state.kit_history.append({"role": "assistant", "content": intro2})
    st.session_state.kit_history.append({"role": "assistant", "content": intro3})
    st.session_state.kit_history.append({"role": "assistant", "content": intro4})

    st.session_state.greetings = not st.session_state.greetings  # Flag to prevent retyping
    st.button('Start!', on_click=conversation_started)

st.session_state.question_index += 1 

col1, col2, col3 = st.columns([5, 1, 1])

# Buttons 
example_prompts = [
    "BACK :material/arrow_back:",
    "HELP :material/help:",
]

example_prompts_help = [
    "Return to the previous question.",
    "Get a list of commands for what KIT can help you with.",
]

def clear_form_keys():
    # Clear condition form keys
    for family_member in st.session_state.family_members:
        for condition in condition_categories.get(st.session_state.current_condition_category, []):
            condition_key = f"condition_{family_member}_{condition}"
            if condition_key in st.session_state:
                del st.session_state[condition_key]

    # Clear family member selection keys
    for family_member in options:
        family_key = f"family_member_{family_member}"
        if family_key in st.session_state:
            del st.session_state[family_key]

    # Clear any form key counters
    if 'form_key_counter' in st.session_state:
        del st.session_state['form_key_counter']

# Function to handle interruption
def handle_interruption():
    kit = st.chat_message('assistant', avatar='🤖')
    clear_form_keys()  # Clear the form keys when interrupted

    if st.session_state.step > 0:
        kit.write_stream(stream_data("Getting back to our family history chat:", 0.5))
        handle_question()  # Re-show the last unanswered question
    else:
        kit.write_stream(stream_data("Getting back to our family history chat:", 0.5))
        st.button('Start!', on_click=conversation_started)

button_cols = st.columns(2)
   
# List of family members and condition categories
options = ["Mother", "Father", "Brother", "Sister", "Grandmother", "Grandfather", "Son", "Daughter","Yourself", "Other","None of the above"]
condition_categories = { 
    "cancer": ["Bladder cancer", "Bone cancer", "Blood or soft tissue cancer", "Brain cancer", "Breast cancer", "Cervical cancer", "Colon Rectal cancer", "Endocrine cancer", "Endometrial cancer", "Esophageal cancer", "Eye cancer", "Head and neck (including cancers of the mouth sinuses, nose or throat, not including brain cancer)", "Kidney cancer", "Lung cancer", "Ovarian cancer", "Pancreatic cancer", "Skin cancer", "Stomach cancer", "Thyroid cancer", "Other", "Don't know"],
    "heart and blood conditions": ["Anemia", "Aortic aneurysm", "Atrial fibrilation (a-fib) or atrial flutter (a-flutter)", "Congestive heart failure" "Coronary Artery/Coronary Heart Disease","Heart attack", "Heart valve disease","High blood pressure (hypertension)", "High cholesterol", "Peripheral vascular disease", "Pulmonary embolism or Deep vein thrombosis (DVT)","Sickle cell disease", "Stroke", "Sudden Death", "Other", "Don't Know"]
}


# Function to display chat messages
def display_chat_history():
    for message in st.session_state['chat_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Function to handle form submission
def handle_question_submission(selected_family_members):
    if selected_family_members:
        st.session_state['family_members'] = selected_family_members
        st.session_state.step = 1  # Proceed to asking condition details

# Function to handle questions and form inputs
def handle_question(user_input=None):
    kit = st.chat_message('assistant', avatar='🤖')

    # Step 0: Ask which family members have been diagnosed with the current condition
    if st.session_state.step == 0:
        current_category = list(condition_categories.keys())[st.session_state.condition_index]
        st.session_state.current_condition_category = current_category
        question = f"Which of your family members have been diagnosed with {current_category}?"
        st.session_state.current_question = question

        # Chat message
        kit.markdown(question)

        # Display the form inside the chat message
        with st.form(key='family_form'):
            selected_family_members = []
            for family_member in options:
                if st.checkbox(family_member, key=f"family_member_{family_member}"):
                    selected_family_members.append(family_member)

            # Submit button for the form
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state.chat_history.append((st.session_state.current_question, selected_family_members))

                handle_question_submission(selected_family_members)
                st.rerun()

    # Step 1: Ask for specific conditions for selected family members
    elif st.session_state.step == 1:
        current_family_member = st.session_state.family_members[st.session_state.family_member_index]
        specific_conditions_list = condition_categories[st.session_state.current_condition_category]
        question = f"What specific type of {st.session_state.current_condition_category} has your {current_family_member} been diagnosed with?"
        st.session_state.current_question = question

        kit.markdown(question)

        selected_condition = []
        with st.form(key=f"condition_form_{current_family_member}"):
            for condition in specific_conditions_list:
                if st.checkbox(condition, key=f"condition_{current_family_member}_{condition}"):
                    selected_condition.append(condition)

            submitted = st.form_submit_button("Submit")
            if submitted and selected_condition:
                st.session_state.specific_conditions.append({
                    "family_member": current_family_member,
                    "conditions": selected_condition
                })
                st.session_state.chat_history.append((st.session_state.current_question, [current_family_member, selected_condition]))

                # Move to next family member or next step
                if st.session_state.family_member_index + 1 < len(st.session_state.family_members):
                    st.session_state.family_member_index += 1  # Move to next family member
                else:
                    st.session_state.step = 2  # Proceed to step 2
                st.rerun()

    # Step 2: Ask for follow-up details
    elif st.session_state.step == 2:
        condition_info = st.session_state.specific_conditions[0]
        follow_up_question = f"Please provide more details about {condition_info['conditions'][0]} for {condition_info['family_member']}."
        kit.markdown(follow_up_question)
        st.session_state.current_question = follow_up_question

        # Text input for further details
        additional_info = st.text_input(follow_up_question)
        if additional_info:
            st.session_state.chat_history.append((st.session_state.current_question, additional_info))
            st.session_state.specific_conditions.pop(0)
            
            # Move to the next family member's details or reset to step 0
            if st.session_state.specific_conditions:
                st.rerun()
            else:
                st.session_state.step = 0  # Reset to step 0 for next condition category
                st.session_state.condition_index += 1
                st.session_state.family_member_index = 0

# Display chat history
#display_chat_history()

def format_response(responses):
    #source_documents = responses["source_documents"]
    #source_content = [doc.page_content for doc in source_documents]
    #source_metadata = [doc.metadata for doc in source_documents]

    #used_urls = []

    #result = responses['result']
    result = responses['answer']
    #result += "\n\n"
    #result += "Sources used:"
    #for i in range(len(source_metadata)):
    
     #   if source_metadata[i]['source'] not in used_urls:
      #      result += "\n\n"
       #     result += source_metadata[i]['title']
        #    result += "\n\n"
         #   result += source_metadata[i]['source']
          #  used_urls.append(source_metadata[i]['source'])
            
    return result


# Capture user input from the chat
prompt = st.chat_input('Chat with KIT, ask a question, go BACK, or get HELP')
if prompt:
    
    with st.chat_message("user"):
        st.markdown(prompt)    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        chain = load_chain_with_sources()
        result = chain({"question":prompt})
        response = format_response(result)
        full_response = ""
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.chat_history.append((st.session_state.current_question, prompt))
# Handle the question and form logic
if st.session_state.conversation_started == 1 and st.session_state.condition_index < len(condition_categories):
    handle_question(prompt)


if uploaded_file is not None:
    if uploaded_file.name.endswith('.txt'):
        # To convert to a string based IO for a txt file:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.write(string_data)
    elif uploaded_file.name.endswith('.csv'):
        # Read the CSV
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
    else:
        st.error("Unsupported file format. Please upload a .txt or .csv file.")

chat_container = st.empty()



def click_button():
    st.session_state.clicked = True

if col2.button(example_prompts[0], help=example_prompts_help[0], use_container_width=True) or prompt =='BACK' or prompt == 'back' or prompt =='Back': #BACK
     if st.session_state['question_index'] > 0:
        st.session_state['question_index'] -= 1  # Go to previous question
        clear_form_keys()
        handle_interruption()
elif col3.button(example_prompts[1], help=example_prompts_help[1],use_container_width=True) or prompt =='HELP' or prompt == 'help' or prompt == 'Help':
    if st.session_state['question_index'] > 0:
        clear_form_keys()
        kit = st.chat_message('assistant', avatar='🤖')
        kit.write("Please use the buttons to select relevant question options or type your answers in the text input field. Click submit to move forward in the conversation.")
        kit.write("Type or press buttons for the following commands:\n\n :red[BACK ]: Return to the previous question \n\n :red[HELP ]: Get a list of commands for what KIT can help you with.")
        kit.write("At any time in the conversation, you may use the text input to ask KIT a clarifying question about conditions or what details you should provide about your family history if you're ever unsure. KIT was designed to answer questions about health and family history, retrieving information from a database of consumer health sources and definitions. KIT will not be able to respond to questions outside this scope.")
        handle_interruption()
    else:
        kit.write("Please use the buttons to select relevant question options or type your answers in the text input field. Click submit to move forward in the conversation.")
        kit.write("Type or press buttons for the following commands:\n\n :red[BACK ]: Return to the previous question \n\n :red[HELP ]: Get a list of commands for what KIT can help you with.")
        kit.write("At any time in the conversation, you may use the text input to ask KIT a clarifying question about conditions or what details you should provide about your family history if you're ever unsure. KIT was designed to answer questions about health and family history, retrieving information from a database of consumer health sources and definitions. KIT will not be able to respond to questions outside this scope.")

# Function to summarize responses using OpenAI API
def summarize_responses(chat_history):
    client = OpenAI(api_key= api_key)
    # Prepare the prompt for summarization
    #st.write(chat_history)
    # Prompt for summarization
    prompt = (
        "Please summarize the user's responses related to what family members have what conditions at a 6th-grade reading level. Do not state you are summarizing at a sixth grade reading level. Please focus on telling them what other details they would need to provide to be eligible for genetic testing and what they should follow up with their family and doctor about. :\n"
         +str(chat_history)+
        "\n\nSummarize these responses in a table format."
    )
    
    try:
        # Call OpenAI API using the correct client method
        completion = client.chat.completions.create(  # or `` based on your implementation
            model="gpt-4o-mini",  
            
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the generated summary
        summary = completion.choices[0].message.content  # Accessing the response correctly
        return summary

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Error in summarization."

if st.session_state.condition_index >= 2:
    st.write("You have completed the survey.")
    
    # Summarize the responses
    summary = summarize_responses(st.session_state.chat_history)
    
    # Display the summary
    st.write(summary)

