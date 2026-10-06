from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 1-2 lines")
detailed_prompt = ChatPromptTemplate.from_template("Explain {topic} in detail")

# Runs both branches concurrently and returns both results in one dict.
chain = RunnableParallel(
    short=short_prompt | model | parser,
    detailed=detailed_prompt | model | parser,
)

if __name__ == "__main__":
    result = chain.invoke({"topic": "machine learning"})
    print("SHORT:", result["short"])
    print("DETAILED:", result["detailed"])
