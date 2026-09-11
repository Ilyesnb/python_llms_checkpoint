from ingest import load_documents, split_documents

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


documents = load_documents()
chunks = split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

vectorstore.save_local("faiss_index")

print("FAISS index created successfully.")