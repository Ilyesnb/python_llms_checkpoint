from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents():
    documents = []

    for file in Path("data/documents").glob("*.txt"):
        text = file.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file.name}
            )
        )

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n---")
        print(chunk.page_content)
        print(chunk.metadata)