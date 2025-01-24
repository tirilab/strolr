# **Strolr: A Retrieval-Augmented Generation Chatbot for Pregnancy Health Information**
Repo for “Strolr: An LLM-enabled Chatbot to Support Pregnant Women’s Quick and Easy Information Seeking from Trustworthy Sources.”
AMIA National Symposium 2024, San Francisco, CA. 

Strolr is a chatbot designed to help pregnant individuals access trustworthy, verified health information quickly and effectively. By summarizing and delivering health information from verified resources, Strolr supports users of varying health literacy levels by providing clear, concise, and easy-to-understand responses.

---

## **Table of Contents**
1. [Solution Overview](#solution-overview)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)


---

## **Solution Overview**

### **Objective**
The primary goal of Strolr is to connect pregnant individuals to trustworthy health resources in an easy-to-access and user-friendly manner. While resources like the CDC, FDA, and HHS provide valuable information, their content may not always be easy to navigate, query, or understand for users with varying health literacy levels.

### **Key Requirements**
- Accessible to users with varying health literacy levels.
- Simple, jargon-free responses in a 6th-grade reading level.
- Quick access to health information from trusted organizations.
- Clear citations for retrieved information.

### **How It Works**
1. **Resource Compilation**: 
   - Strolr uses a curated list of over 100 verified health resources from trusted U.S. government agencies, such as:
     - Centers for Disease Control and Prevention (CDC)
     - U.S. Food and Drug Administration (FDA)
     - U.S. Department of Health and Human Services (HHS)

2. **Data Storage and Retrieval**:
   - Document embeddings are generated from the compiled resources and stored in an **Amazon RDS for PostgreSQL** database hosted on an **AWS Virtual Private Cloud (VPC)**.

3. **Chat Workflow**:
   - Users interact with a Streamlit web app to input their queries.
   - Documents relevant to the query are retrieved using **vector similarity** matching.
   - The retrieved documents are summarized using **LangChain** and **OpenAI's GPT-3.5-turbo model**.
   - Strolr’s responses:
     - Summarize the content at a basic health literacy level.
     - Cite all retrieved documents.
     - Encourage users to consult healthcare providers for queries beyond the scope of the chatbot.

---

## **Features**
- **Trustworthy Sources**: Strolr restricts responses to verified pregnancy health resources.
- **Customizable Responses**: Summarizes content at a 6th-grade reading level for easy understanding.
- **Citations**: All responses include references to the source documents.
- **Fallback Guidance**: For queries beyond the chatbot’s scope, users are encouraged to consult healthcare providers.
- **Scalable Backend**: Powered by AWS for scalable and secure data handling.

---

## **Setup Instructions**

### **Prerequisites**
1. Install Python (version >= 3.8).
2. Install PostgreSQL and set up an RDS instance.
3. Obtain API keys for OpenAI.

### **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/strolr.git
   cd strolr

### **File Navigation**
1. **url_resources.csv** - This is a list of resources curated for Strolr's pilot database. The URLs are provided for the relational database.
2. **lanchain.ipynb** - This python notebook shows the process of loading the URLs from the curated list, chunks the documents, embeds them (need OpenAI key here), creates a vector store (Docker), and adds the documents to the vector store.
  a. A **docs.pkl** file was created to save the documents. You can use **vectordb.py** to only embed and add the data from the vector store by loading the docs.pkl file.
3. **strolr_amia.py** - This file contains the necessary functions for running the full Strolr Streamlit app, with connections to your database and OpenAI key required to run.
4. **stror_bot.svg** and **LOGO_FINAL.png** - Logos for app.
5. **script.js**, **index.html**, **styles.css** - Files for informational resources link of Strolr Streamlit app. 
   

