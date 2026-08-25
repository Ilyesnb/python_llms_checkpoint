import json
from datetime import date

from pydantic import BaseModel, Field, field_validator

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_ollama import ChatOllama


# 1. PYDANTIC MODEL

class DocumentMetadata(BaseModel):

    title: str = Field(
        ...,
        description="Title of the document"
    )

    author: str = Field(
        ...,
        description="Author of the document"
    )

    publication_date: str = Field(
        ...,
        description="Publication date in YYYY-MM-DD format"
    )

    keywords: list[str] = Field(
        ...,
        description="List of keywords related to the document"
    )

    document_type: str = Field(
        ...,
        description="Type of document, for example report or article"
    )

    # Validate publication date

    @field_validator("publication_date")
    @classmethod
    def validate_publication_date(cls, value):

        try:
            date.fromisoformat(value)

        except ValueError:
            raise ValueError(
                "publication_date must use YYYY-MM-DD format"
            )

        return value


# 2. JSON OUTPUT PARSER

parser = JsonOutputParser(
    pydantic_object=DocumentMetadata
)



# 3. PROMPT TEMPLATE

prompt = PromptTemplate(
    template="""
You are a document metadata extraction system.

Extract the following metadata from the document:

- title
- author
- publication_date
- keywords
- document_type

Rules:

1. Return ONLY valid JSON.
2. Follow the provided schema exactly.
3. publication_date MUST use YYYY-MM-DD.
4. keywords MUST be a list of strings.
5. Do not add additional fields.
6. Do not invent information.
7. If information is missing, use an empty string.

{format_instructions}

DOCUMENT:

{document_text}
""",

    input_variables=["document_text"],

    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)



# 4. OLLAMA MODEL


model = ChatOllama(
    model="llama3.2:latest",
    temperature=0
)


# 5. UNSTRUCTURED DOCUMENTS


documents = [

    """
    This report, 'AI Trends 2025', was written by Dr. Sarah Lee
    and published on 2025-05-01.

    It covers topics like artificial intelligence,
    deep learning, and ethics in AI.
    """,

    """
    The article 'Climate Change and Agriculture' by John Smith
    was released on 2024-09-15.

    It discusses sustainability, farming practices,
    and environmental policy.
    """
]


# 6. PROCESS DOCUMENTS

structured_metadata = []


for index, document in enumerate(documents, start=1):

    print()
    print("=" * 60)
    print(f"Processing document {index}")
    print("=" * 60)

    try:

        # Create prompt

        prompt_text = prompt.format(
            document_text=document
        )

        # Send document to Ollama

        response = model.invoke(prompt_text)

        print("\nRaw response:")
        print(response.content)

        # Parse JSON

        parsed_data = parser.parse(
            response.content
        )

        print("\nParsed data:")
        print(parsed_data)

        # Validate with Pydantic

        metadata = DocumentMetadata(
            **parsed_data
        )

        print("\nPydantic validation:")
        print("✓ Validation successful")

        # Convert Pydantic object to dictionary

        metadata_dict = metadata.model_dump()

        # Add to final dataset

        structured_metadata.append(
            metadata_dict
        )

        print("\nValidated metadata:")

        print(
            json.dumps(
                metadata_dict,
                indent=4,
                ensure_ascii=False
            )
        )

        print("\n✓ Document successfully processed")

    except Exception as error:

        print("\n✗ Error while processing document:")

        print(
            type(error).__name__,
            ":",
            error
        )


# 7. FINAL STRUCTURED DATASET

print()
print("=" * 60)
print("FINAL STRUCTURED DATASET")
print("=" * 60)

print(
    json.dumps(
        structured_metadata,
        indent=4,
        ensure_ascii=False
    )
)


# 8. SAVE JSON DATASET

with open(
    "output.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        structured_metadata,
        file,
        indent=4,
        ensure_ascii=False
    )


print()
print("✓ Dataset saved to output.json")