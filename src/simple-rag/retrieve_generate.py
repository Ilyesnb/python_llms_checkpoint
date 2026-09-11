from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


question = input("Question: ")

documents = retriever.invoke(question)

context = "\n\n".join(
    f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
    for doc in documents
)

prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.

If the answer is not in the context, say:
"I don't know based on the provided documents."

Always mention the source of your answer.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)

print("\nAnswer:")
print(response.content)

print("\nSources:")
for doc in documents:
    print("-", doc.metadata["source"])