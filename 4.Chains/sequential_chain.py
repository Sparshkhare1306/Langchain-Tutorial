from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
load_dotenv()

# main_model= "mistralai/Mistral-7B-Instruct-v0.2"
# backup_model= "google/gemma-3-27b-it"
# both these models are good for conversational tasks

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="text-generation"
)

model= ChatHuggingFace(llm= llm)

parser= StrOutputParser()
template1= PromptTemplate(
    template= "give me a detailed information about the topic {topic} \n",
    input_variables= ["topic"]
    
)

template2= PromptTemplate(
    template= "give me a summary in 5 lines about the paragraph {paragraph} \n",
    input_variables= ["paragraph"])

chain= template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic': "Inland taipan"})

print(result)
chain.get_graph().print_ascii()