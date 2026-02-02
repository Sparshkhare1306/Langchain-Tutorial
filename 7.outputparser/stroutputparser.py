from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
load_dotenv()

model1, model2= "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational"
)


model= ChatHuggingFace(llm= llm)


# prompt1 :detailed report
template1= PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables= ["topic"]
)

# prompt2 : summary
template2= PromptTemplate(
    template= "write a 5 line summary of the following text: {text}",
    input_variables= ["text"]
)

prompt1 = template1.invoke({"topic": "Black holes"})
result1= model.invoke(prompt1)
#print("Detailed Report:\n", result1.content)

prompt2 = template2.invoke({"text": result1.content})
result2= model.invoke(prompt2)
print("\nSummary:\n", result2.content)
