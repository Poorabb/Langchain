# Use for static webpages. for more JS heavy webpages, use seleniumUrlLoader

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
prompt = PromptTemplate(
    template = " Answer the question {ques} from the following text: {text}",
    input_variables = ["question","text"]
)

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

url = "https://en.wikipedia.org/wiki/Glass_(2019_film)"
loader = WebBaseLoader(url)

doc = loader.load()

chain = prompt | model | parser  

print(chain.invoke({"ques":"who made this movie?`","text":doc[0].page_content}))
