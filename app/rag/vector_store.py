from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Load the document
# --------------------------------------------------

loader = TextLoader(
    "data/documents/ai_notes.txt",
    encoding="utf-8"
)

documents = loader.load()

print("Loaded documents:", len(documents))


# --------------------------------------------------
# 2. Split the document
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(documents)

print("Created chunks:", len(chunks))


# --------------------------------------------------
# 3. Create Embedding Model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Create Chroma Vector Database
# --------------------------------------------------

vector_store = Chroma(
    collection_name="ai_assistant_documents",
    persist_directory="data/chroma_db",
    embedding_function=embedding_model,
)


# --------------------------------------------------
# 5. Add documents to Chroma
# --------------------------------------------------

vector_store.add_documents(chunks)


# --------------------------------------------------
# 6. Check stored documents
# --------------------------------------------------

count = vector_store._collection.count()

print("Documents stored in Chroma:", count)

print("\nVector database created successfully.")