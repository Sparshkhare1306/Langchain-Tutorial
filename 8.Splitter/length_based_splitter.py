from langchain_text_splitters import CharacterTextSplitter



text= "LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more."

splitter= CharacterTextSplitter(
    chunk_size= 5,
    chunk_overlap= 0,
    separator=''
)
result= splitter.split_text(text)

print(result)