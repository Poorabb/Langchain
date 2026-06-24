from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "6 RAG\\Document loaders",
    glob = "*.pdf", # Every pdf 
    loader_cls= PyPDFLoader # Define which loader for indivisual files
)

docs = loader.load()

print(len(docs))