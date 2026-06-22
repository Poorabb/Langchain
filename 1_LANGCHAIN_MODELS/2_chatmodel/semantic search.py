from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

query = input("BAKO BSDK: ")
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

documents= [
    "Sachin Tendulkar is Legendary batsman with 100 international centuries.",
    "Virat Kohli is One of the best modern batsmen known for consistency and aggression.",
    "MS Dhoni is Former Indian captain who led India to three ICC trophies.",
    "Rohit Sharma is  Nicknamed Hitman and holds the highest ODI score of 264.",
    "Kapil Dev is All-rounder who captained India to its first World Cup win in 1983."
]


doc_embed=embedding.embed_documents(documents)

query_embed=embedding.embed_query(query)

score=list(cosine_similarity([query_embed],doc_embed)[0])

print(documents[score.index(max(score))])