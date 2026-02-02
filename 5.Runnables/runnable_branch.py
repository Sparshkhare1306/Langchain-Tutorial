from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda, RunnableBranch, RunnablePassthrough
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
prompt1= PromptTemplate(
    template= "give me a detailed info about {topic}",
    input_variables= ['topic']
)

prompt2= PromptTemplate(
    template= "give me a short 5 line summary of the paragraph \n {paragraph}",
    input_variables= ["paragraph"]
)

first_chain= RunnableSequence(prompt1 | model | parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>200, RunnableSequence(prompt2 | model |parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(first_chain | branch_chain)

print(final_chain.invoke({'topic': "Inland Taipan"}))