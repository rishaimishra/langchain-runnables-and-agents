from langchain_core.tools import tool


@tool
def get_greeting(name: str) -> str:
    """Generate a greeting message for a user."""
    return f"Hello {name}, welcome to the AI world"


if __name__ == "__main__":
    result = get_greeting.invoke({"name": "Rishav"})
    print(result)
    print(get_greeting.name)
    print(get_greeting.description)
    print(get_greeting.args)
