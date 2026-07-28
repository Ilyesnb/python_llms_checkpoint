from langchain_ollama import ChatOllama
# 1. Initialize the local Llama model via Ollama
# temperature controls creativity: 0.0 is deterministic/factual, 1.0 is creative
llm = ChatOllama(
 model="llama3.2",
 temperature=0.7,
)
# 2. Define a simple string question
question = "Explain how a solar panel converts sunlight into electricity."
print(f"Asking Llama: '{question}'\n")
print("Thinking...\n")
# 3. Directly invoke the model (no prompt templates or chains needed)
response = llm.invoke(question)
# 4. Print the result
# The model returns an AIMessage object, so we use .content to get just the clean text
print("--- AI Response ---")
print(response.content)