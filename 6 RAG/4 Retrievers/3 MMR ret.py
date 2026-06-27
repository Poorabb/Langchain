# MMR- Maximal marginal relevance - Brings diversity in selected answers of relevance.

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone()

index = pc.Index("sample")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

ret = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2,"lambda_mult": 0.5}
)

result = ret.invoke("who are the captains?")
print(result)