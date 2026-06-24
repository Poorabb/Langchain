from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("C:\\Users\\poora\\Desktop\\work\\Langchain\\6 RAG\\1 Document loaders\\poems.pdf")

doc = loader.load()
splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap =  0,
    separator = '' 
)

print(doc[0].page_content)

# result = splitter.split_text(doc[0].page_content)
# print("\n\n", result)

result = splitter.split_documents(doc)
print(result)