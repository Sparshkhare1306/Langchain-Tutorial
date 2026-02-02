from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

# main_model= "mistralai/Mistral-7B-Instruct-v0.2"
# backup_model= "google/gemma-3-27b-it"
# both these models are good for conversational tasks

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="text-generation"
)

model= ChatHuggingFace(llm= llm)
parser= JsonOutputParser()
template= PromptTemplate(
    template= "give me 5 facts about {topic} \n {format_instructions}",
    input_variables= ["topic"],
    partial_variables= {'format_instructions': parser.get_format_instructions()}
)

chain= template | model | parser

result = chain.invoke({"topic": "Dr. A.P.J. Abdul Kalam"})

print(result)  # {'name': 'Alice', 'age': 30, 'city': 'New York'}