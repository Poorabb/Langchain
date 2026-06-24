# Use when mostly text pdf 

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("6 RAG\\Document loaders\\poems.pdf")

docs=loader.load()
print((docs)[0].page_content)

''' 
Other loaders with their better use cases:

PDFPlumberLoader  - PDFs with tables 
UnstructuredPDFLoader / AmazonTextractPDFLoader - Scanned/Image PDFs 
PyMuPDFLoader - Need layout and image data
CSVLoader - Load CSVs

'''
