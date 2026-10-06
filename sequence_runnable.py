from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Explain {topic} in simple words")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# The basic pipe: each stage's output becomes the next stage's input.
chain = prompt | model | parser

if __name__ == "__main__":
    result = chain.invoke({"topic": "machine learning"})
    print(result)
