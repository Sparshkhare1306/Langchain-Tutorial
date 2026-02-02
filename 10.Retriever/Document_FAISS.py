from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

docs=[
    Document(page_content="LangChain is a framework for developing applications powered by language models."),
    Document(page_content="It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="LangChain provides integrations with various vector stores including FAISS.")
]

# initialize the embedding model
embedding_model= OpenAIEmbeddings()

# create the FAISS vector store from the documents and embeddings
vectorstore= FAISS.from_documents(documents= docs, embedding= embedding_model)

# Enable retrieval with MMR
retriever= vectorstore.as_retriever(search_type= "mmr", search_kwargs= {"k": 3, "lambda_mult":0.5})

# use the retriever to get relevant documents for a query
query= "What is LangChain?"
result= retriever.invoke(query)


for i, doc in enumerate(result):
    print(f"Result {i+1}:")
    print(doc.page_content)



