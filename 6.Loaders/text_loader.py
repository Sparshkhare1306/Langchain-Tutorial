from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm= HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
)


model= ChatHuggingFace(llm= llm)

loader= TextLoader('/Users/sparsh_khare/Desktop/Langchain Tutorial/ghibli.txt', encoding= 'utf-8')
docs= loader.load()
prompt= PromptTemplate(
    template= "Give me a small summary of this text: \n {text}",
    input_variables= ["text"]
)

parser= StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))