from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
# import streamlit as st
from sqlalchemy import create_engine
import json
import re
from typing import Tuple, List,Dict,Any
import sqlite3
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
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD') 
host = os.getenv('DB_HOST')
database = os.getenv('DB_NAME')


# Create a SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
database_store = 'investment_tci_prompt'
engine2 = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database_store}")

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)

def _strip(text: str) -> str:
    return text.strip()


from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime



Base = declarative_base()



class UserQuery(Base):
    __tablename__ = 'user_queries'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_query = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

# Create the table
Base.metadata.create_all(engine2)



def store_user_query(query: str, engine):
    if not query:  # Check if query is None or an empty string
        print("Error: Query cannot be empty.")
        return
    
    session = sessionmaker(bind=engine)()
    new_query = UserQuery(user_query=query, timestamp=datetime.now())
    session.add(new_query)
    session.commit()
    session.close()



def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.This dataset consists of the global financial flows from both public and private sectors directed to tackle plastic pollution. The dataset covers multiple data points for each financial flow, including the time period, the name, institution type, and geography of both the flow provider and recipient, the application of the financial flow, and the flow amount based on multiple types of financial flow, such as loan, equity, or grant.
    Based on the table schema below, write a SQL query that would answer the user's question.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)to the database.
  
    Instruction:
    -To filter 'public finance' and 'private finance', always use the 'category' column. You can filter 'private finance' in also 'sub_category'.
    -To filter 'Domestic Finance', 'Development Finance','Private Finance' and 'MEA Finance', always use the 'sub_category' column.
    -When asked of questions by type of funding always refer to 'sub_category' column. 
    -To filter 'Circular Pathways', 'Non-circular Pathways' and 'Supporting Pathways', always use the 'pathways' column.
    -There is columns like ft_grant. don't get 'SUM(ft_grant) AS grant' like this, use another name instead.
    -To filter 'Value Recovery', 'Circular Design and Production', 'Circular Use', 'Clean-up', 'Digital Mapping', 'Incineration', 'Managed Landfilling', 'Operational Platforms', 'Other Services', 'Plastic Waste-to-Energy', 'Plastic Waste-to-Fuel', 'Recovery for controlled disposal', 'Research and Development' and 'Training/ Capacity Building/ Education', always use the 'archetype' column.
    -When asked about program/project description as a general question, you have to use 'sector_name' to get the answer.
    -To filter 'Africa', 'Asia', 'Europe', 'Latin America And The Caribbean', 'Oceania' and 'North America', always use the 'region' column.
    -To filter 'Multi Donor National', 'Multilateral' and 'Multi Donor Regional', always use the 'fund_type' column.
    -To filter by fund name such as  'Adaptation for Smallholder Agriculture Programme (ASAP)', 'Adaptation Fund (AF)', 'Amazon Fund', 'Forest Investment Program (FIP)', 'Global Climate Change Alliance (GCCA)', and 'Global Environment Facility (GEF4)', always use the 'fund_name' column.
    -Unique value of 'sids', 'lldc', 'fcs' and 'ida' are '0' and '1'
    -To check IDA eligible countries need to filter always from 'ida' column. Ids value '1' means eligible and '0' means not eligible. 
    -To filter 'Total funding', 'Deal value','total capital' and 'total spend' 'amount of private investment', 'investment' or 'commitment', always use the 'financial_flow' column.
    -There are 7 types of ODA such as  'ocean_oda', 'sustainable_ocean_oda', 'land_based_oda', 'plastic_oda','plastic_specific_oda','solid_waste_oda', 'wastewater_oda', when ask of ODA as a general question, you have to get all 1 values for all 7 columns always and get the answer.

    For example:
    1. Question: Based on the last 5 years, trend of funding towards plastic pollution, what do you expect in the next 3 years?
       SQL Query: SELECT pathway, SUM(financial_flow) AS total_funding FROM finances WHERE pathway IN ('Circular Pathways', 'Non-circular Pathways', 'Supporting Pathways') GROUP BY pathway;

    2. Question: What is the split of funding between circular pathways and non-circular pathways?
       SQL Query: SELECT pathways, SUM(financial_flow) AS total_funding FROM finances WHERE pathways IN ('Circular Pathways', 'Non-circular Pathways') GROUP BY pathways;

    3. Question: Which country is the biggest official development assistance provider for tackling plastic pollution in 2021?
       SQL Query: SELECT provider_country, SUM(financial_flow) AS total_funding FROM finances WHERE year = 2021 GROUP BY provider_country ORDER BY total_funding DESC LIMIT 1;

    Your turn:
    
    Question: {question}
    SQL Query:
    """
  
  prompt = ChatPromptTemplate.from_template(template)
  
  model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4")

  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model
    | StrOutputParser()
  )

from datetime import datetime
current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company analyzing financial flows for plastic pollution. Based on the provided SQL query and its results, create a clear and accurate response.
    
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    
    Original SQL Query: {query}
    SQL Response: {response}
    User Question: {question}

    Important Instructions:
    1. Use EXACTLY the same SQL query that was generated - DON'T MODIFY IT (warning!!!!)
    2. Base your answer ONLY on the SQL query results - do not use external knowledge
    3. For financial values: 
       - Do NOT convert or simplify the numbers
       - Keep ALL digits exactly as shown in the SQL output
       - Just add '$' prefix and ' million' suffix
       - Example: if SQL shows 19798.00, write as '$19,798.0 million' (NOT as '$19.8 million')
       - All numbers should have exactly one decimal place, no matter how they appear in the SQL output
    4. For time-based data, describe clear trends
    5. When comparing values, provide relative differences,
    6. Don't mention about technical things like 'Based on SQL result' like
    7. Give your answer in one sentence
    
    Visualization Guidelines - ONLY choose one if needed:
    1. Use 'line_chart' for:
       - Time series data
       - Trend analysis over periods
       Example format:
       [
           {{"year": 2020, "funding": 1000000}},
           {{"year": 2021, "funding": 1200000}}
       ]

    2. Use 'stack_bar_chart' for:
       - Comparing parts of a whole
       - Multiple categories over time
       Example format:
       [
           {{"category": "Type A", "value1": 100, "value2": 200}},
           {{"category": "Type B", "value1": 150, "value2": 250}}
       ]

    3. Use 'bar_chart' for:
       - Simple category comparisons
       - Single metric analysis
       - Distribution across categories
       Example format:
       [
           {{"region": "Asia", "funding": 500000}},
           {{"region": "Europe", "funding": 700000}}
       ]

    Your response should follow this format:
    graph_needed: "yes" or "no"
    graph_type: one of ['line_chart', 'stack_bar_chart', 'bar_chart', 'text']
    data_array: [your data array if graph is needed]
    text_answer: Your detailed explanation

    Remember: Focus on accuracy and clarity in your response.
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4-0125-preview")
    
    try:
        # Get and execute the SQL query
        query = sql_chain.invoke({
            "question": user_query,
            "chat_history": chat_history,
        })
        print(query)
        
        sql_response = db.run(query)
        
        # If no data found
        if not sql_response:
            return """
            graph_needed: no
            graph_type: text
            text_answer: No data found for your query. Please try refining your search criteria.
            """
        
        # Generate the response
        chain = (
            prompt 
            | model 
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "schema": db.get_table_info(),
            "chat_history": chat_history,
            "query": query,
            "response": sql_response,
            "question": user_query
        })
        
        return response
        
    except Exception as e:
        return {
        'provider':'bot',
        'datetime':current_timestamp,
        'type':'error',
        'content': 'Unfortunately I am unable to provide a response for that. Could you send me the prompt again?',
        'data':None
        }
        


    # 4. Use 'pie_chart' for:
    #    - Showing proportions of a whole
    #    - Distribution across categories
    #    Example format:
    #    [
    #        {{"name": "Category A", "value": 1000000}},
    #        {{"name": "Category B", "value": 2000000}}
    #    ]



# Function to extract fields using regex
import json
import re
def extract_response_data(result):
    # Updated regex patterns
    graph_needed_pattern = r'graph_needed:\s*"?(yes|no|[\w\s]+)"?'
    graph_type_pattern = r'graph_type:\s*(\S.*)'
    data_array_pattern = r'\[\s*(.*?)\s*\]'

    # Extract fields
    graph_needed = re.search(graph_needed_pattern, result)
    graph_type = re.search(graph_type_pattern, result)
    data_array = re.search(data_array_pattern, result, re.DOTALL)

    # Extract and clean values
    graph_needed_value = graph_needed.group(1) if graph_needed else None
    graph_type_value = graph_type.group(1).strip().strip('"') if graph_type else None
    data_array_str = data_array.group(1) if data_array else None

    text_pattern = r'text_answer:\s*(\S.*)'
    text_output = re.search(text_pattern, result)
    text_str = text_output.group(1).strip().strip('"') if text_output else None

    print("=========== data passed to plot the graph =============")
    print(graph_needed_value)
    print(graph_type_value)
    print(data_array_str)
    print("=======================================================")
    print(text_str)

    if data_array_str:
        data_string = f"[{data_array_str}]"
        try:
            data_array_value = json.loads(data_string)
        except json.JSONDecodeError:
            print("Error decoding JSON from data_array.")
            data_array_value = None
    else:
        data_array_value = None

    # Process the data to a dynamic format
    if data_array_value and isinstance(data_array_value, list) and len(data_array_value) > 0:
        # Use the first entry to determine label and dataset keys dynamically
        first_entry = data_array_value[0]
        
        # Use any key as a label key if it appears in all entries
        possible_keys = list(first_entry.keys())
        
        # Choose the first available key as label key and use the rest for dataset values
        label_key = possible_keys[0]
        data_keys = possible_keys[1:] if len(possible_keys) > 1 else []
        
        # Extract labels and datasets
        labels = [item.get(label_key, "N/A") for item in data_array_value]
        datasets = [
            tuple(item.get(key, None) for key in data_keys)
            for item in data_array_value
        ]
        
        formatted_data = {
            "labels": labels,
            "datasets": datasets,
            "legend": False
        }
    else:
        formatted_data = {"error": "Data array is empty or not in expected format."}

    return graph_needed_value, graph_type_value, formatted_data, text_str


# from flask import Flask, request, jsonify
# from langchain_core.messages import HumanMessage, AIMessage
# from datetime import datetime

# app = Flask(__name__)

# # In-memory chat history for demo purposes
# chat_history = [{"role": "AI", "content": "Hello! I'm a SQL assistant. Ask me anything about your database."}]

# @app.route('/api/chat', methods=['POST'])
# def chat():
#     try:
#         if not request.is_json:
#             return jsonify({'error': 'Content-Type must be application/json'}), 400

#         data = request.get_json()
#         if 'message' not in data:
#             return jsonify({'error': 'Message field is required'}), 400

#         user_query = data['message'].strip()
#         if not user_query:
#             return jsonify({'error': 'Query cannot be empty.'}), 400

#         # Prepare chat history for get_response function
#         formatted_history = []
#         for msg in chat_history:
#             role = msg.get("role")
#             content = msg.get("content")
#             if content is None:
#                 return jsonify({"error": "Message content missing in chat history"}), 500
#             if role == "user":
#                 formatted_history.append(HumanMessage(content=content))
#             elif role == "AI":
#                 formatted_history.append(AIMessage(content=content))

#         # Generate response with existing function
#         response = get_response(user_query, db, formatted_history)
#         graph_needed, graph_type, data_array, text_answer = extract_response_data(response)

#         # Append messages to chat history
#         chat_history.append({"role": "user", "content": user_query})
#         chat_history.append({"role": "AI", "content": text_answer})

#         # Keep only last N messages to limit chat history size
#         max_history = 10
#         if len(chat_history) > max_history * 2:
#             del chat_history[:-max_history * 2]

#         current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         # Return structured JSON response
#         return jsonify({
#             'provider':'bot',
#             'datetime':current_timestamp,
#             'type':graph_type,
#             'content': text_answer,
#             'data': data_array
#         }), 200

#     except Exception as e:
#         return jsonify({
#             'provider':'bot',
#             'datetime':current_timestamp,
#             'type':'error',
#             'content': 'Unfortunately I am unable to provide a response for that. Could you send me the prompt again?',
#             'data':None
#         }), 500

# @app.route('/api/clear-history', methods=['POST'])
# def clear_history():
#     global chat_history
#     chat_history = [{"role": "AI", "content": "Hello! I'm a SQL assistant. Ask me anything about your database."}]
#     return jsonify({"message": "Chat history cleared"}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False)



from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.secret_key = 'Abc123@'

# In-memory chat history for demo purposes
chat_history = [{"role": "AI", "content": "Hello! I'm a SQL assistant. Ask me anything about your database."}]

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        if 'message' not in data:
            return jsonify({'error': 'Message field is required'}), 400

        user_query = data['message'].strip()
        if not user_query:
            return jsonify({'error': 'Query cannot be empty.'}), 400

        # Prepare chat history for get_response function
        formatted_history = []
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            if content is None:
                return jsonify({"error": "Message content missing in chat history"}), 500
            if role == "user":
                formatted_history.append(HumanMessage(content=content))
            elif role == "AI":
                formatted_history.append(AIMessage(content=content))

        # Generate response with existing function
        response = get_response(user_query, db, formatted_history)
        graph_needed, graph_type, data_array, text_answer = extract_response_data(response)

        # Append messages to chat history
        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "AI", "content": text_answer})

        # Keep only last N messages to limit chat history size
        max_history = 10
        if len(chat_history) > max_history * 2:
            del chat_history[:-max_history * 2]

        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Return structured JSON response
        return jsonify({
            'provider':'bot',
            'datetime':current_timestamp,
            'type':graph_type,
            'content': text_answer,
            'data': data_array
        }), 200

    except Exception as e:
        return jsonify({
            'provider':'bot',
            'datetime':current_timestamp,
            'type':'error',
            'content': 'Unfortunately I am unable to provide a response for that. Could you send me the prompt again?',
            'data':None
        }), 500

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = [{"role": "AI", "content": "Hello! I'm a SQL assistant. Ask me anything about your database."}]
    return jsonify({"message": "Chat history cleared"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
