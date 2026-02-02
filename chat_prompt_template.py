from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

chat_template = PromptTemplate([
    SystemMessage(content="You are a helpful {domain} expert"),
    HumanMessage(content="Tell me about {topic} in detail")
])

prompt= chat_template.invoke({
    'domain': 'cricket',
    'topic': 'LBW'
})

print(prompt)