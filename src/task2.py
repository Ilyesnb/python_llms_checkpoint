from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
llm = ChatOllama(model="llama3.2", temperature=0.7)

# 1. Define Prompt Template with placeholders like {country} and {user_question}
prompt = ChatPromptTemplate.from_messages([
    ("system", "you are an expert in computer science and programming, and you will answer questions about {topic}."),
    ("human", "question about programming: {user_question}"),
])

# 2. Format Messages Directly (Pass the dynamic arguments here)
formatted_messages = prompt.format_messages(
    topic="computer science",
    user_question="What is the difference between a stack and a queue in data structures?"
    
)

# 3. Invoke Model directly
response = llm.invoke(formatted_messages)
print(response.content)
