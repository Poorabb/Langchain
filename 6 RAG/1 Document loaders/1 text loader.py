from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

prompt = PromptTemplate(
    template=" Explain the meaning behind this poem: {text}",
    input_variables = ['text']
)

loader = TextLoader('C:\\Users\\poora\\Desktop\\work\\Langchain\\6 RAG\\Document loaders\\poem.txt')

docs = loader.load()

chain  = prompt | model | parser

result = chain.invoke({'text':docs[0].page_content})

print(f"Poem: {docs[0].page_content} , \n\n Meaning: {result}")

