from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


model = ChatOllama(
    model="llama3.2",
    temperature=0.7
)


# Conversation buffer
conversation_history = []


# Assistant role
system_message = SystemMessage(
    content="""
    You are a helpful travel assistant.
    You recommend budget-friendly trips.
    You remember information from previous messages.
    """
)


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message to memory
    conversation_history.append(
        HumanMessage(content=user_input)
    )

    # Send the complete conversation to the model
    messages = [system_message] + conversation_history

    # Get assistant response
    response = model.invoke(messages)

    print("Assistant:", response.content)

    # Add assistant response to memory
    conversation_history.append(
        AIMessage(content=response.content)
    )