from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://192.168.1.182:11434",
    model="llama3.1",
    temperature=0,
    # other params...
)

from langchain_core.messages import AIMessage

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)