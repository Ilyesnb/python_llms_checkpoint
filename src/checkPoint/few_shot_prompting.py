from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Load the model
model = ChatOllama(model="llama3.2", temperature=0.7)

# Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an {topic} assistant."),

    # Example 1
    ("human", 'Email: "Dinner at 8 tonight? I\'ll bring the wine."'),
    ("ai", "Personal"),

    # Example 2
    ("human", 'Email: "You have won a free iPhone! Click here to claim your prize."'),
    ("ai", "Spam"),

    # Example 3
    ("human", 'Email: "The Q2 financial report is due by end of day tomorrow."'),
    ("ai", "Work"),

    # New email to classify
    ("human", 'Email: "{email}"')
])

# Format the prompt
formatted_messages = prompt.format_messages(
    topic="email classification",
    email="Are you free for lunch this weekend?"
)

# Invoke the model
response = model.invoke(formatted_messages)

# Print the result
print(response.content)