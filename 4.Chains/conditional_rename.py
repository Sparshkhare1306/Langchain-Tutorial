from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

from typing import Literal
load_dotenv()
parser= StrOutputParser()
llm= HuggingFaceEndpoint(repo_id= "deepseek-ai/DeepSeek-V3.2-Exp")
model= ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative']= Field (description="The sentiment of the feedback")

parser2= PydanticOutputParser(pydantic_object= Feedback)



prompt1= PromptTemplate(
    template= "classify the sentiment of the following feedback as Positive or negative \n {feedback} \n {format_instructions}"
    "You MUST respond with exactly one of: Positive or Negative.\n"
    "If the sentiment is neutral or mixed, choose the closer option.\n\n",
    input_variables= ["feedback"],
    partial_variables= {'format_instructions': parser2.get_format_instructions()}
)

classifier_chain= prompt1 | model | parser2

prompt2= PromptTemplate(
    template= "write an appropriate response to this positive feedback \n {feedback} \n",
    input_variables= ["feedback"]
)

prompt3= PromptTemplate(
    template= "write an appropriate response to this negative feedback \n {feedback} \n",
    input_variables= ["feedback"]
)

branch_chain = RunnableBranch(
    # we send a tuple in this, it follws the format of giving a tuple which is (condition, chain to be executed if condition is met or not met) 
    (lambda x: x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")

)

chain= classifier_chain | branch_chain

print(chain.invoke({'feedback': "This is a great phone."}))

chain.get_graph().print_ascii()