from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain.core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'classify the given feedback into positive or negative sentiment: {feedback}. {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser 
