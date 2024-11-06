import getpass
import os
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
import pickle

from langchain_openai import OpenAI
from openai import OpenAI
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
connection = "postgresql+psycopg://langchain:langchain@strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com:5432/postgres"
collection_name = "strolr_docs"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
#query = "What should I do if I have a post-partum depression?"
query = "Is it safe for my unborn baby if I eat raw fish during pregnancy?"
#similar = vector_store.similarity_search_with_score(query, k=3)

def load_chain_with_sources():
    
    embeddings = OpenAIEmbeddings()
    
    # CONNECT TO RDS
    connection = "postgresql+psycopg://langchain:langchain@strolrdb.c348i082m9zo.us-east-2.rds.amazonaws.com:5432/postgres"
    collection_name = "strolr_docs"
    store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,)
    retriever = store.as_retriever(search_type="similarity_score_threshold", search_kwargs = {"k":3, "score_threshold":0.8})
    llm = ChatOpenAI(temperature = 0.8, model = "gpt-4o-mini")
    


    # Create memory 'chat_history' 
    #memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages = True)
    #memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", output_key='answer', return_messages = True)

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
        If the answer is not in the {context}, say that you don't know in a kind way or give them a suggestion on a different question to ask.
        Do your best to understand typos, casing, and framing of questions. 
	    Do not return sources if you responded with I don't know.
       
        {context}
       """

    # Create the Conversational Chain
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{input}"),
    ]
)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Set up the RAG chain
    chain = create_retrieval_chain(retriever, question_answer_chain)
    

    # Invoke the RAG chain with the question
    return chain 


chain = load_chain_with_sources()

formatted_query = {'input': query}
result = chain.invoke(formatted_query)
print(result)