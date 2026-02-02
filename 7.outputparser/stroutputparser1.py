from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# main_model= "mistralai/Mistral-7B-Instruct-v0.2"
# backup_model= "google/gemma-3-27b-it"
# both these models are good for conversational tasks

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="text-generation"
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

parser= StrOutputParser()

chain= template1 | model | parser | template2 | model | parser 

result = chain.invoke({"topic": "Black holes"})

print(result)