from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("C:\\Users\\poora\\Desktop\\work\\Langchain\\6 RAG\\1 Document loaders\\poems.pdf")

doc = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap =  0,
)


# result = splitter.split_text(doc[0].page_content)
# print("\n\n", result)

result = splitter.split_documents(doc)
print(result[0].page_content)


'''

To use this method for different languages like python or markdown,

from langchain_text_splitter import language

splitter = RecursiveCharacterTextSplitter.from_language(
    language = language.PYTHON / any language
    chunk_size = 100,
    chunk_overlap =  0,
)
'''