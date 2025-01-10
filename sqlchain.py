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

print(sqlite3.sqlite_version)

import chromadb
# from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, PrivateAttr
from typing import List


from sqlalchemy import Column, Integer, String, Text,DateTime
from sqlalchemy.orm import declarative_base,sessionmaker

import os
import pandas as pd
from sqlalchemy import Column, Integer, Text, DateTime
import json
from openai import OpenAI
from datetime import datetime


# Load OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD') 
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')


##################################################################################################
###########################RAG part###############################################################

# chroma_client = chromadb.HttpClient(host='3.110.107.185', port=8000)
chroma_client = chromadb.HttpClient(host='3.110.90.22', port=8000)
chroma_collection = chroma_client.get_collection("fonterra")


class ChromaDBRetriever(BaseRetriever, BaseModel):
    """Custom retriever for ChromaDB that properly implements Pydantic BaseModel"""
    _collection: any = PrivateAttr()
    top_k: int = 3

    def __init__(self, **data):
        super().__init__(**data)
        self._collection = chroma_collection

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self._collection.query(
            query_texts=[query],
            n_results=self.top_k
        )
        return [Document(page_content=doc) for doc in results['documents'][0]]

# Initialize the retriever
retriever = ChromaDBRetriever()


llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o")

def rag_response(question: str) -> str:
    template = """You are an assistant for question-answering tasks. 
    Use the following context to answer the question. If you don't know the answer, just say that you don't know.

    Context: {context}
    Question: {question}

    For Example: 
     - What are the National level scores of each KPI?
     - What are 4 KPIs that are considered in perfect store classification?
     - What is perfect store?

    Provide a clear and direct answer without any JSON formatting or special characters.
    """

    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": prompt,
        }
    )

    try:
        result = qa_chain.invoke({"query": question})
        # The result is returned directly as a string in newer versions of LangChain
        if isinstance(result, str):
            return result
        # If result is a dict, get the answer from the appropriate key
        elif isinstance(result, dict):
            # Try different possible keys that might contain the answer
            if 'result' in result:
                return result['result']
            elif 'answer' in result:
                return result['answer']
            elif 'output' in result:
                return result['output']
            else:
                # If we can't find the answer in any expected key, return the full result as string
                return str(result)
        else:
            # For any other type, convert to string
            return str(result)
    except Exception as e:
        print(f"Error during chain execution: {str(e)}")
        print(f"Result type: {type(result)}")
        print(f"Result content: {result}")
        return "I apologize, but I encountered an error while processing your question. Please try asking in a different way."

#End RAG part
############################################################################################################



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
    -When ask about 'target achievement rate', you need to get total visit in particular month then divide this value with 225 and get percentage value(value should be INT)

    For example:
    Question: How many area sales managers are there?
    SQL Query: SELECT COUNT(DISTINCT `Area_Sales_Manager`) FROM fonterra_pilot;
    Question: How many stores are assigned to MILAN KODIKARA?
    SQL Query: SELECT SUM(Total_Base_of_80%_Contribution_outlets_assigned) AS total_stores FROM visit_summary WHERE Area_Sales_Manager = 'MILAN KODIKARA';

    
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



##################function calling#######################
def execute_sql_query(user_query: str, db):
    """Execute database queries for numerical data and statistics."""
    chat_history = [
        AIMessage(content="Hello, I am an AI chatbot specialized in giving fonterra Area sales managers sales details"),
        HumanMessage(content=user_query)
    ]

    response = get_response(user_query, db, chat_history)

    if response:
        return {
            'content': response,
        }
    else:
        return {
            'content': 'Unfortunately I am unable to provide a response for that. Could you send me the prompt again?',
        }


# def retrieve_from_document(user_query: str):
#     """Retrieve definitional, conceptual, and contextual information from documents."""
#     response = rag_response(user_query)
#     # Get the current UTC timestamp
#     if response:
#         return {
#             'content':response,
#         }
#     # return response

def retrieve_from_document(user_query: str) -> str:
    """Retrieve definitional, conceptual, and contextual information from documents."""
    print(f"Starting document retrieval for query: {user_query}")
    try:
        response = rag_response(user_query)
        print(f"RAG response type: {type(response)}")
        print(f"RAG response content: {response}")
        return str(response)
    except Exception as e:
        print(f"Error in retrieve_from_document: {str(e)}")
        return "I encountered an error while searching the documents. Please try again."


# def retrieve_from_document(query):
#     results = retriever.get_relevant_documents(query)
#     # Combine all retrieved document content into a single string
#     return "\n".join([doc.page_content for doc in results])


# Function
functions = [
    {
        "name": "execute_sql_query",
        "description": (
            "Use this function for questions about numerical data, statistics, and store performance metrics. "
            "Examples: How many stores are assigned to Lakmal? How many stores were visited by Lakmal in October?"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The query about numerical data, performance metrics, or store statistics.",
                },
            },
            "required": ["user_query"],
        },
    },
    {
        "name": "retrieve_from_document",
        "description": (
            "Use this function for questions about definitions, concepts, methodologies, and classifications related to Perfect Store. "
            "Examples: What is a Perfect Store? What are the KPIs used in Perfect Store classification? what are the National level scores of each KPI? "
            "What does MCL compliance mean?"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "The query about definitions, concepts, or methodology from the Perfect Store document.",
                },
            },
            "required": ["user_query"],
        },
    },
]


def get_chatbot_response_with_history(user_message: str, chat_history: list):
    """Modified main function to handle user queries with chat history."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Convert chat history to the format OpenAI expects
    messages = [
        {
            "role": "system",
            "content": """You are a specialized assistant for Perfect Store analysis. For queries, follow these instructions:
            - For numerical data, statistics, performance metrics, or store-related analytics (e.g., store visited information, store assign information): use the function execute_sql_query.
            - For definitions, concepts, methodologies, or classifications related to Perfect Store (e.g., what is MCL compliance, KPIs, or Perfect Store criteria, National level scores): use the function retrieve_from_document.
            - Always use one of these functions—do not answer directly.
            - For follow-up questions, use the context from previous messages to understand what the user is asking.
            - If the question cannot be answered using the database or document, respond with 'I'm unable to find the required information.'."""
        }
    ]
    
    # Add chat history
    for msg in chat_history:
        if isinstance(msg, (AIMessage, HumanMessage)):
            messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": str(msg.content)  # Ensure content is string
            })
    
    # Add current message
    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model="gpt-4-0613",
            messages=messages,
            functions=functions,
            function_call="auto"
        )

        response = completion.choices[0].message

        if response.function_call:
            function_name = response.function_call.name
            function_args = json.loads(response.function_call.arguments)
            
            print(f"Called function: {function_name}")
            print(f"Function arguments: {function_args}")
            
            if function_name == "execute_sql_query":
                result = execute_sql_query(function_args["user_query"], db)
                print(f"SQL result type: {type(result)}")
                return str(result)
            elif function_name == "retrieve_from_document":
                result = retrieve_from_document(function_args["user_query"])
                print(f"Document result type: {type(result)}")
                return str(result)
                
            return "I couldn't process your query. Please try again."
    except Exception as e:
        print(f"Error stack trace:")
        import traceback
        print(traceback.format_exc())
        return f"An error occurred: {str(e)}"
##########################endfunctioncalling#################





# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#       AIMessage(content="Hello! I'm your assistant. Ask me anything about your area sales manager."),
#     ]

# load_dotenv()

# st.set_page_config(page_title="fontera_ai", page_icon=":speech_balloon:")
# # st.image("download.jpeg", width=200) 

# # st.title("Fonterra Assistant!🤖")
# st.markdown("<h1 style='text-align: left; color: white; margin-left: 15px; '>Fonterra Assistant!🤖</h1>", unsafe_allow_html=True)

# st.session_state.db = db
# st.markdown("""
#     <style>
               
#     .stApp {
#         background-color: #2E5077;
#         color: white;
#     }
            
#     .logo-img {
#         position: fixed;
#         top: 10px;
#         left: 5px;
#         margin: 60px 10px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Display the image
# st.markdown(
#     f"""
#     <img src="data:image/jpeg;base64,{base64.b64encode(open('download.jpeg', 'rb').read()).decode()}" class="logo-img" width="100">
#     """,
#     unsafe_allow_html=True
# )


# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         with st.chat_message("AI"):
#             st.markdown(f"<span style='color: white;'>{message.content}</span>", unsafe_allow_html=True)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.markdown(f"<span style='color: white;'>{message.content}</span>", unsafe_allow_html=True)



# user_query = st.chat_input("Type a message...")
# if user_query is not None and user_query.strip() != "":
#     # Append the user's message to chat history
#     st.session_state.chat_history.append(HumanMessage(content=user_query))
#     with st.chat_message("user"):
#         st.markdown(f"<span style='color: white;'>{user_query}</span>", unsafe_allow_html=True)

#     # Generate AI response
#     response = get_response(user_query, st.session_state.db, st.session_state.chat_history)

#     if response:
#         # Append the AI response to chat history
#         st.session_state.chat_history.append(AIMessage(content=response))
#         with st.chat_message("assistant"):
#             st.markdown(f"<span style='color: white;'>{response}</span>", unsafe_allow_html=True)
#     else:
#         # Fallback response if no valid answer is generated
#         fallback_message = "I couldn't generate a valid response. Please try again."
#         st.session_state.chat_history.append(AIMessage(content=fallback_message))
#         with st.chat_message("assistant"):
#             st.markdown(f"<span style='color: white;'>{fallback_message}</span>", unsafe_allow_html=True)



def main():
    """Main function to run the Streamlit Perfect Store Assistant"""
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your assistant. Ask me anything about the Perfect Store."),
        ]
    
    # Load environment variables
    load_dotenv()
    
    # Setup page configuration
    st.set_page_config(page_title="Perfect Store Assistant", page_icon=":speech_balloon:")
    
    # Add custom CSS
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
    
    # Add header and logo
    st.markdown("<h1 style='text-align: left; color: white; margin-left: 15px;'>Perfect Store Assistant!🤖</h1>", 
                unsafe_allow_html=True)
    
    try:
        # Display the logo
        with open('download.jpeg', 'rb') as f:
            logo_data = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <img src="data:image/jpeg;base64,{logo_data}" class="logo-img" width="100">
                """,
                unsafe_allow_html=True
            )
    except FileNotFoundError:
        st.warning("Logo file not found. Please ensure 'download.jpeg' exists in the correct directory.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
            st.markdown(f"<span style='color: white;'>{message.content}</span>", unsafe_allow_html=True)
    
    # Handle user input
    user_query = st.chat_input("Type a message...")
    
    if user_query and user_query.strip():
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(f"<span style='color: white;'>{user_query}</span>", unsafe_allow_html=True)
        
        try:
            response_text = get_chatbot_response_with_history(user_query, st.session_state.chat_history)
            
            # Ensure response_text is a string
            if response_text is not None:
                response_text = str(response_text)
            else:
                response_text = "No response received"
            
            # Create AI message with string content
            ai_message = AIMessage(content=response_text)
            st.session_state.chat_history.append(ai_message)
            
            with st.chat_message("assistant"):
                st.markdown(f"<span style='color: white;'>{response_text}</span>", unsafe_allow_html=True)
                
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.session_state.chat_history.append(AIMessage(content=error_message))
            with st.chat_message("assistant"):
                st.markdown(f"<span style='color: white;'>{error_message}</span>", unsafe_allow_html=True)
            print(f"Error in chat response: {str(e)}")

if __name__ == "__main__":
    main()

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         AIMessage(content="Hello! I'm your assistant. Ask me anything about the Perfect Store."),
#     ]

# load_dotenv()

# st.set_page_config(page_title="Perfect Store Assistant", page_icon=":speech_balloon:")

# # Customize page header
# st.markdown("<h1 style='text-align: left; color: white; margin-left: 15px;'>Perfect Store Assistant!🤖</h1>", unsafe_allow_html=True)

# # Customize the app background and styles
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #2E5077;
#         color: white;
#     }
#     .logo-img {
#         position: fixed;
#         top: 10px;
#         left: 5px;
#         margin: 60px 10px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Display the logo
# st.markdown(
#     f"""
#     <img src="data:image/jpeg;base64,{base64.b64encode(open('download.jpeg', 'rb').read()).decode()}" class="logo-img" width="100">
#     """,
#     unsafe_allow_html=True
# )

# # Display chat history
# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         with st.chat_message("AI"):
#             st.markdown(f"<span style='color: white;'>{message.content}</span>", unsafe_allow_html=True)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("Human"):
#             st.markdown(f"<span style='color: white;'>{message.content}</span>", unsafe_allow_html=True)

# # Capture user input
# user_query = st.chat_input("Type a message...")
# if user_query is not None and user_query.strip() != "":
#     # Append user's message to chat history
#     st.session_state.chat_history.append(HumanMessage(content=user_query))
#     with st.chat_message("user"):
#         st.markdown(f"<span style='color: white;'>{user_query}</span>", unsafe_allow_html=True)

#     # Generate AI response
#     try:
#     # Get the response
#         response = get_chatbot_response_with_history(user_query, st.session_state.chat_history)

#         # Check if the response is an instance of AIMessage
#         if isinstance(response, AIMessage):
#             # Access the content attribute directly
#             ai_response = response.content
#         elif isinstance(response, str):  # If the response is plain text
#             ai_response = response
#         else:
#             # Handle unexpected types
#             ai_response = "I couldn't process your query. Please try again."

#         # Append to chat history and display the response
#         st.session_state.chat_history.append(AIMessage(content=ai_response))
#         with st.chat_message("assistant"):
#             st.markdown(f"<span style='color: white;'>{ai_response}</span>", unsafe_allow_html=True)

#     except Exception as e:
#         # Catch and display errors
#         error_message = f"An error occurred: {str(e)}"
#         st.session_state.chat_history.append(AIMessage(content=error_message))
#         with st.chat_message("assistant"):
#             st.markdown(f"<span style='color: white;'>{error_message}</span>", unsafe_allow_html=True)
