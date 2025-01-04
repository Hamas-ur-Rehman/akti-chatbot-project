import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from weather import get_weather_data



@tool
def weather(location_name):
    """
    Get the weather data for the given location by passing location_name varaible.
    """
    return get_weather_data(location_name)

tools = [weather]

PROMPT = """You are a chatbot that speaks roman urdu with english words .
You will answer every question in urdu no matter what.

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