from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
model = ChatOllama(model="llama3.2", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an {topic} assistant."),
    ("human","""{instruction}Email:{email}Answer with only one word""")
])
formatted_prompt = prompt.format_messages(
    topic ="email classification",
    instruction="Classify emails as Work, Personal, or Spam.",
    email="Don't forget the team meeting at 2 PM." 
)
response = model.invoke(formatted_prompt)
print(response.content)