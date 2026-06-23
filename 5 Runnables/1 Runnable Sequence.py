from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt1 = PromptTemplate(
    template = "write a joke about the following topic: {topic}",
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser)

print(chain.invoke({'topic':'cricket'}))

# Since sequence was used too much, new syntax was introduced using | operator -> prompt1 | model | parser . Thats it ! 