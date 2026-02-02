from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = OpenAIEmbeddings(model= "text-embedding-3-small", dimensions= 300)
documents= ["Apple is red and green in color.",
            "The capital of France is Paris.",
            "Tell me more about yourself.",
            "My name is Anthony Garcia.",
            "Virat Kohli is a great cricketer."
            ]


query= 'Tell me about a cricketer'

documents_embeddings= embeddings.embed_documents(documents)
query_embedding= embeddings.embed_query(query)

similarities= cosine_similarity([query_embedding], documents_embeddings) # this is always a 2-D list

print("Similarities: ", similarities)
