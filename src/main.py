from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(model="llama3.2", temperature=0.7)

template = ("what is the longest montaine in the world? Please answer in a concise manner.")

prompt = ChatPromptTemplate.from_template(template)

response = model.invoke(prompt.format())

print(response.content)