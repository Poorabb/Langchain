from langchain_google_genai import GoogleGenerativeAI
from dotenv import  load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-3-flash-preview")
resylt=llm.invoke("Hi bro how are you?")
print(resylt)