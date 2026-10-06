from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator. Return only the code, no prose."),
    ("human", "{topic}"),
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain code in simple terms."),
    ("human", "Explain the following code in simple words:\n{code}"),
])

# Stage 1: generate the code.
generate = code_prompt | model | parser

# Stage 2: RunnablePassthrough carries the generated code forward unchanged
# into "code", while the same value is reshaped into {"code": ...} for the
# explanation branch's prompt template.
explain = RunnableParallel(
    code=RunnablePassthrough(),
    explanation=RunnableLambda(lambda code: {"code": code}) | explain_prompt | model | parser,
)

chain = generate | explain

if __name__ == "__main__":
    result = chain.invoke({"topic": "write a Python function that checks if a string is a palindrome"})
    print("CODE:\n", result["code"])
    print("\nEXPLANATION:\n", result["explanation"])
