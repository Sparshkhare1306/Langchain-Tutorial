from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import Embeddings
from dotenv import load_dotenv
load_dotenv()

embedding= OpenAIEmbeddings(model= 'text-embedding-3-small', dimensions= 32)

result= embedding.embed_query("New Delhi is the capital of India")

print(str(result))