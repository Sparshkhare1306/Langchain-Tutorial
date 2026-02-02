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

prompt1= PromptTemplate(
    template= "tell me detailed info about {info}",
    input_variables= ['info']
)

prompt2= PromptTemplate(
    template=(
        "Extract 5 interesting facts from the text below.\n"
        "Respond ONLY in valid JSON.\n"
        "{format_instructions}\n\n"
        "Text:\n{input}"
    ),
    input_variables=["input"],
    partial_variables={
        "format_instructions": JsonOutputParser().get_format_instructions()
    }
)
json_parser = JsonOutputParser()
str_parser = StrOutputParser()

chain = RunnableSequence( prompt1 | model | str_parser | prompt2 | model | json_parser)

print(chain.invoke({'info': "Black Mamba"}))