"""
Binds a tool to a model and manually walks through the tool-call round
trip: ask -> model requests a tool call -> run the tool -> feed the
result back -> model gives the final answer.
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text."""
    return len(text)


TOOLS = {"get_text_length": get_text_length}


def main():
    model = ChatOpenAI(model="gpt-4o-mini")
    model_with_tools = model.bind_tools([get_text_length])

    messages = [HumanMessage("How many characters are in 'LangChain'?")]
    response = model_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        call = response.tool_calls[0]
        tool_result = TOOLS[call["name"]].invoke(call)
        messages.append(tool_result)
        response = model_with_tools.invoke(messages)

    print(response.content)


if __name__ == "__main__":
    main()
