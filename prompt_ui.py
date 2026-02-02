from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import PromptTemplate
load_dotenv()

st.header("Chat with OpenAI's Chat Model")


paper_input= st.selectbox('Select research paper name', ['Attention Is All You Need', 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', 'GPT-3: Language Models are Few-Shot Learners'])

style_input= st.selectbox('Select writing style', ['Formal', 'Informal', 'Humorous', 'Technical', 'Conversational', 'Beginner-Friendly'])

length_input= st.selectbox('Select length of summary', ['Short (1-2 paragraphs)', 'Medium (3-4 paragraphs)', 'Long (5+ paragraphs)'])


#template
template= PromptTemplate(
    input_variables= ['paper_input', 'style_input', 'length_input'],
    template= 'Summarize the research paper titled "{paper_input}" in a {style_input} style and {length_input} length.'
)

prompt= template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

#user_input= st.text_input('Enter prompt')
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    max_new_tokens=256,
)
model = ChatHuggingFace(llm=llm)
if st.button('Click here'):
    result = model.invoke(prompt)
    st.write(result.content)