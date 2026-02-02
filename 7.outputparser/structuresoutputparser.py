"""
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema

load_dotenv()

# main_model= "mistralai/Mistral-7B-Instruct-v0.2"
# backup_model= "google/gemma-3-27b-it"
# both these models are good for conversational tasks

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="text-generation"
)

model= ChatHuggingFace(llm= llm)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
    
]
parser= StructuresOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    template= "give me 3 facts about {topic} \n {format_instructions}",
    input_variables= ["topic"],
    partial_variables= {'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "Dr. A.P.J. Abdul Kalam"})

print(result)
"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Define schema using Pydantic
class Facts(BaseModel):
    fact_1: str = Field(description="Fact 1 about the topic")
    fact_2: str = Field(description="Fact 2 about the topic")
    fact_3: str = Field(description="Fact 3 about the topic")

parser = JsonOutputParser(pydantic_object=Facts)

template = PromptTemplate(
    template="Give me 3 facts about {topic}\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({"topic": "Dr. A.P.J. Abdul Kalam"})
print(result)
