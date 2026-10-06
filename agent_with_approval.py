"""
A tool-calling agent with a human-in-the-loop middleware: every tool call
is shown to the user for approval before it actually runs. Giving an
agent tools is the easy part - controlling when it's allowed to act on
them is what makes it safe to leave running unattended.
"""
import os
from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url, timeout=10)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {city}: {desc}, {temp}°C"


@tool
def count_characters(text: str) -> int:
    """Count the number of characters in a piece of text."""
    return len(text)


@wrap_tool_call
def human_approval(request, handler):
    """Ask for approval in the terminal before every tool call runs."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.strip().lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"],
        )

    return handler(request)


def build_agent():
    model = ChatOpenAI(model="gpt-4o-mini")
    return create_agent(
        model,
        tools=[get_weather, count_characters],
        system_prompt="You are a helpful assistant with access to tools.",
        middleware=[human_approval],
    )


def main():
    agent = build_agent()
    print("Agent ready | type 'exit' to quit")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print("Agent:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
