# Branch -> Conditionals in langchain Runnables 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on the topic: {topic}",
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

Generator_chain = prompt1 | model | parser 

prompt2 = PromptTemplate(
    template = "Shorten the text into a summary of less than 300 words: {text}",
    input_variables=['text']
)
# Conditional Runnable 

Check_limit = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough()
)

chain = Generator_chain | Check_limit 

print(chain.invoke({"topic":"Recent developments in AI"}))