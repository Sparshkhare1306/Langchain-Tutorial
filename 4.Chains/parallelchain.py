from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
with open("text.txt", "r", encoding="utf-8") as f:
    text_data = f.read()
load_dotenv()

# main_model= "mistralai/Mistral-7B-Instruct-v0.2"
# backup_model= "google/gemma-3-27b-it"
# both these models are good for conversational tasks

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="text-generation"
)
llm2= HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)


model1= ChatHuggingFace(llm= llm1)
model2= ChatHuggingFace(llm= llm2)

prompt1= PromptTemplate(
    template= "generate short and simple notes from the following text  \n {text}",
    input_variables= ["text"]
    
)

prompt2= PromptTemplate(
    template= "Generate 5 short question/answers from the following text \n {text} \n",
    input_variables= ["text"])


prompt3 = PromptTemplate(
    template=(
        "You are given NOTES and a QUIZ.\n"
        "You MUST include BOTH sections in the final output.\n\n"
        "Format the final document EXACTLY like this:\n\n"
        "### NOTES\n"
        "{notes}\n\n"
        "### QUIZ (Question–Answer)\n"
        "{quiz}\n\n"
        "DO NOT summarize, omit, or rewrite content.\n"
        "Preserve all questions and answers."
    ),
    input_variables=["notes", "quiz"]
)

parser= StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser 

chain = parallel_chain | merge_chain

result = chain.invoke({'text': text_data})
print(result)