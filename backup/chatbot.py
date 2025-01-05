import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from weather import get_weather_data
from info_db import pdf_database,csv_database,web_database



@tool
def weather(location_name):
    """
    Get the weather data for the given location by passing location_name varaible. This indo is from Open-Meteo API.
    """
    return get_weather_data(location_name)

@tool
def pdf_db(question):
    """
    Get the data from pdf database which contains data about EMR documents.
    """
    return pdf_database(question)

@tool
def csv_db(question):
    """
    Get the data from csv database which contains data about insurance values for people with different conditions.
    """
    return csv_database(question)

@tool
def web_db(question):
    """
    Get the data from web database which contains data from Dawn News and BBC.
    """
    return web_database(question)


tools = [weather,pdf_db,csv_db,web_db]

PROMPT = """You are a chatbot that speaks roman urdu with english words .
You will answer every question in urdu no matter what.

Make sure to add a tag line like this information was provided by Dawn News or BBC or our weather API or csv dataset.

User: What is the capital of Pakistan?
Chatbot: pakistan ka darulhakoomat islamabad hai.
"""

model = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools)
messages = [SystemMessage(PROMPT)]

def chatbot(question):
    messages.append(HumanMessage(question))
    response = model.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
                if tool_call['name'] == "weather":
                    print(tool_call['args'])
                    weather_data = weather(tool_call['args']['location_name'])
                    tool_output = ToolMessage(weather_data,tool_call_id=tool_call["id"])
                    messages.append(ToolMessage(weather(tool_call['args']['location_name']),tool_call_id=tool_call["id"]))

                if tool_call['name'] == "pdf_db":
                    print(tool_call['args'])
                    pdf_data = pdf_db(tool_call['args']['question'])
                    tool_output = ToolMessage(pdf_data,tool_call_id=tool_call["id"])
                    messages.append(ToolMessage(pdf_db(tool_call['args']['question']),tool_call_id=tool_call["id"]))

                if tool_call['name'] == "csv_db":
                    print(tool_call['args'])
                    csv_data = csv_db(tool_call['args']['question'])
                    tool_output = ToolMessage(csv_data,tool_call_id=tool_call["id"])
                    messages.append(ToolMessage(csv_db(tool_call['args']['question']),tool_call_id=tool_call["id"]))

                if tool_call['name'] == "web_db":
                    print(tool_call['args'])
                    web_data = web_db(tool_call['args']['question'])
                    tool_output = ToolMessage(web_data,tool_call_id=tool_call["id"])
                    messages.append(ToolMessage(web_db(tool_call['args']['question']),tool_call_id=tool_call["id"]))

                    
        response = model.invoke(messages)
        messages.append(response)
    
    return response.content
                    



# while True:
#     question = input("Human: ")
#     messages.append(HumanMessage(question))
#     if question.lower() == "exit":
#         break
#     response = model.invoke(messages)
#     messages.append(response)

#     if response.tool_calls:
#         for tool_call in response.tool_calls:
#                 if tool_call['name'] == "weather":
#                     print(tool_call['args'])
#                     weather_data = weather(tool_call['args']['location_name'])
#                     tool_output = ToolMessage(weather_data,tool_call_id=tool_call["id"])
#                     messages.append(ToolMessage(weather(tool_call['args']['location_name']),tool_call_id=tool_call["id"]))

#         response = model.invoke(messages)
#         messages.append(response)
                    
#     print("AI: ",response.content)
#     print()