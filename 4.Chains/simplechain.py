from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="text-generation"
)

model= ChatHuggingFace(llm= llm)
parser= StrOutputParser()
template= PromptTemplate(
    template= "give me 5 facts about {topic} \n ",
    input_variables= ['topic']

)



chain = template | model | parser 

result = chain.invoke({'topic': "Inland taipan"})

chain.get_graph().print_ascii()