from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a detailed report about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a quiz on the following topic: {topic}',
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template = 'Merge the 2 pieces of text and present together properly. notes -> {notes} and quiz -> {quiz}',
    input_variables = ['notes','quiz']
)

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes':prompt1 | model | parser, 
    'quiz': prompt2 | model | parser
})

merger_chain = prompt3 | model | parser

chain = parallel_chain | merger_chain 

result = chain.invoke({'topic':'cricket'})
print(result)

chain.get_graph().draw_ascii()