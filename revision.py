from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
result=llm.invoke("whats todays news in india")
print(result.text)