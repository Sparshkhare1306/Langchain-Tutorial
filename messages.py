from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv
load_dotenv()

llm= HuggingFaceEndpoint(repo_id= "mistralai/Mistral-7B-Instruct-v0.2")
chat_model = ChatHuggingFace(llm= llm)

messages= [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello, who won the world series in 2020?")
]
result = chat_model.invoke(messages)
messages.append(AIMessage(content= result.content))

print(messages)