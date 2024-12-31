from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
from sqlalchemy import create_engine
import json
import re
from typing import Tuple, List,Dict,Any
import sqlite3
print(sqlite3.sqlite_version)

from sqlalchemy import Column, Integer, String, Text,DateTime
from sqlalchemy.orm import declarative_base,sessionmaker

import os
import pandas as pd
from sqlalchemy import Column, Integer, Text, DateTime
import json
from openai import OpenAI
from datetime import datetime
import base64


# Load OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD') 
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')


# engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
from urllib.parse import quote_plus
encoded_password = quote_plus(password)

# Create connection string
connection_string = f"mysql+pymysql://{user}:{encoded_password}@{host}/{database}"

try:
    engine = create_engine(connection_string)
    # Test the connection
    with engine.connect() as connection:
        print("Database connection successful!")
    db = SQLDatabase(engine)
except Exception as e:
    print(f"Error connecting to database: {str(e)}")


def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    Instruction:
    -To filter 'store','outlets', always use the 'Total_outlets_assigned' column.

    For example:
    Question: How many area sales managers are there?
    SQL Query: SELECT COUNT(DISTINCT `Area_Sales_Manager`) FROM fonterra_pilot;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4-0125-preview")
  model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4")
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model
    | StrOutputParser()
  )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    
    Important Instructions:
    1. Use EXACTLY the same SQL query that was generated - DON'T MODIFY IT (warning!!!!)
    2. Base your answer ONLY on the SQL query results - do not use external knowledge
    3. When comparing values, provide relative differences,
    4. Don't mention about technical things like 'Based on SQL result','Based on table' like"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4-0125-preview")
  llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4-0125-preview")
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })



if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm your assistant. Ask me anything about your area sales manager."),
    ]

load_dotenv()

st.set_page_config(page_title="fontera_ai", page_icon=":speech_balloon:")
# st.image("download.jpeg", width=200) 

# st.title("Fonterra Assistant!🤖")
st.markdown("<h1 style='text-align: center; color: white;'>Fonterra Assistant!🤖</h1>", unsafe_allow_html=True)

st.session_state.db = db
st.markdown("""
    <style>
               
    .stApp {
        background-color: #2E5077;
        color: white;
    }
            
    .logo-img {
        position: fixed;
        top: 10px;
        left: 5px;
        margin: 60px 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Display the image
st.markdown(
    f"""
    <img src="data:image/jpeg;base64,{base64.b64encode(open('download.jpeg', 'rb').read()).decode()}" class="logo-img" width="100">
    """,
    unsafe_allow_html=True
)

# st.logo("download.jpeg")
# st.image("download.jpeg", width=200) 

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)


user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    # Append the user's message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate AI response
    response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
    if response:
        # Append the AI response to chat history
        st.session_state.chat_history.append(AIMessage(content=response))
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        # Fallback response if no valid answer is generated
        fallback_message = "I couldn't generate a valid response. Please try again."
        st.session_state.chat_history.append(AIMessage(content=fallback_message))
        with st.chat_message("assistant"):
            st.markdown(fallback_message)


