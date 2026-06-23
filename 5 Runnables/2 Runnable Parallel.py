from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

TweetPrompt = PromptTemplate(
    template="Prepare a well styled tweet under 20 awords on the topic: {topic} ",
    input_variables=['topic']
)

LinkedInPrompt = PromptTemplate(
    template="Write a detailed linked in post on the topic: {topic}",
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet": TweetPrompt | model | parser,
    "Post": LinkedInPrompt | model | parser
})

print(parallel_chain.invoke({'topic':'recent AI developments'})['tweet']) # prints only tweet 