from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
llm = ChatOllama(model="llama3.2", temperature=0.7)

# 1. Define Prompt Template with placeholders like {country} and {user_question}
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that knows about {topic}."),
    ("human", "What is the capital of {country}?"),
])
country= input("Enter a country name: ")
# 2. Format Messages Directly (Pass the dynamic arguments here)
formatted_messages = prompt.format_messages(
    topic="geography",
    country=country
)

# 3. Invoke Model directly
response = llm.invoke(formatted_messages)
print(response.content)
