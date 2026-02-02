from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()
llm= HuggingFaceEndpoint(repo_id= "mistralai/Mistral-7B-Instruct-v0.2")

chat_model = ChatHuggingFace(llm= llm)
chat_history= [
    SystemMessage(content="You are a helpful ai assistant.")
]
while True:
    user_input= input('You:')
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break
    result = chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('Bot:', result.content)

print("Chat history:", chat_history)
