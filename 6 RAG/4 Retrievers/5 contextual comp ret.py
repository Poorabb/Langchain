from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

pc = Pinecone()

index = pc.Index("data")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

model = ChatGoogleGenerativeAI(model = "gemini-3-flash-preview")


vector_store = PineconeVectorStore(index=index, embedding=embeddings)

ret = vector_store.as_retriever(
    search_kwargs={"k": 2,}
)
comp = LLMChainExtractor.from_llm(model)


comp_ret = ContextualCompressionRetriever(
    base_compressor=comp,
    base_retriever=ret
)

query = "what is crucial for cellular repair?"
print(comp_ret.invoke(query))


