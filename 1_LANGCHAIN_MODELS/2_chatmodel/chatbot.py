
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview",temperature=0.7,max_output_tokens=1000,thinking_budget=100)
result=model.invoke("What are 7 days in the week")
print(result.text)