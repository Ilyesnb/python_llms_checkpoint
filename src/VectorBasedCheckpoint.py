
# 1. IMPORT LIBRARIES


import faiss
from sentence_transformers import SentenceTransformer
import numpy as np



# 2. PREPARE THE DOCUMENT CORPUS


corpus = [
    "Artificial Intelligence is transforming industries such as manufacturing, healthcare, and finance.",
    
    "FAISS is a library developed for efficient similarity search of dense vectors.",
    
    "Machine learning models require high-quality data for training and accurate predictions.",
    
    "Python is widely used for data science, machine learning, and artificial intelligence.",
    
    "Industrial robots use artificial intelligence to automate manufacturing processes.",
    
    "Computer vision allows machines to analyze and understand images and videos.",
    
    "Predictive maintenance uses machine learning to predict equipment failures before they happen.",
    
    "Natural language processing allows computers to understand and generate human language.",
    
    "Data analysis helps companies make better decisions using information collected from their operations.",
    
    "AI can improve industrial productivity by automating repetitive tasks and optimizing processes."
]

print(f"Number of documents: {len(corpus)}")



# 3. LOAD THE EMBEDDING MODEL


model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. GENERATE EMBEDDINGS


embeddings = model.encode(
    corpus,
    convert_to_numpy=True
)

print("Embedding shape:", embeddings.shape)



# 5. BUILD THE FAISS INDEX


# Dimension of each embedding
d = embeddings.shape[1]

# Create an index using L2 (Euclidean) distance
index = faiss.IndexFlatL2(d)

# Add the document embeddings to the index
index.add(embeddings.astype("float32"))

print("Number of vectors in FAISS:", index.ntotal)



# 6. TEST RETRIEVAL


query = "How can I use AI in industry?"

# Convert the query into an embedding
query_vector = model.encode(
    [query],
    convert_to_numpy=True
)

# FAISS expects float32 vectors
query_vector = query_vector.astype("float32")

# Search for the top 2 most similar documents
D, I = index.search(query_vector, k=2)



# 7. DISPLAY RESULTS


print("\nQuery:")
print(query)

print("\nTop Results:")

for rank, idx in enumerate(I[0], start=1):
    print(f"\nResult {rank}")
    print(f"Distance: {D[0][rank - 1]:.4f}")
    print(f"Document: {corpus[idx]}")