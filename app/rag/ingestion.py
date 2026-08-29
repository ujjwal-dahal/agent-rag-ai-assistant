from langchain_community.document_loaders import TextLoader


# Path to the document
file_path = "data/documents/ai_notes.txt"


# Create the document loader
loader = TextLoader(
    file_path,
    encoding="utf-8"
)


# Load the document
documents = loader.load()


# Display the loaded documents
print("Number of documents:", len(documents))

for document in documents:
    print("\nDocument Content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)