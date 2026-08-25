import ollama


document = """
The Eiffel Tower, located in Paris, France, was constructed between 1887 and 1889
as the entrance arch to the 1889 World’s Fair.

It was designed by Gustave Eiffel’s engineering company and initially faced
criticism from artists and intellectuals who considered it an eyesore.

Today, the Eiffel Tower is one of the most visited monuments in the world,
attracting over 7 million visitors annually.

In 2015, special lighting systems were added to enhance its nighttime appearance
and improve energy efficiency.
"""


# ============================================================
# 1. SPLIT DOCUMENT INTO MULTIPLE CHUNKS
# ============================================================

def split_document(text, chunk_size=30):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


chunks = split_document(document, chunk_size=30)


print("================================")
print("DOCUMENT CHUNKS")
print("================================")

for i, chunk in enumerate(chunks, 1):

    print(f"\n--- CHUNK {i} ---")
    print(chunk)


# ============================================================
# 2. EXTRACT FACTS FROM EACH CHUNK
# ============================================================

def extract_facts(chunk):

    prompt = f"""
Extract the key facts from this text.

Identify:
- Dates
- Places
- People
- Numbers
- Events
- Important entities

Text:
{chunk}

Return only the important facts.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


print("\n\n================================")
print("KEY FACTS")
print("================================")


all_facts = []

for i, chunk in enumerate(chunks, 1):

    print(f"\n--- FACTS FROM CHUNK {i} ---")

    facts = extract_facts(chunk)

    all_facts.append(facts)

    print(facts)


# ============================================================
# 3. GENERATE SUMMARY
# ============================================================

def generate_summary(text):

    prompt = f"""
Write a short summary of the following document.

Keep the most important information.

Document:
{text}
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


summary = generate_summary(document)


print("\n\n================================")
print("SUMMARY")
print("================================")

print(summary)