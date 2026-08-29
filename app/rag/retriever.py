from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Create the same Embedding Model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Load the Chroma Vector Database
# --------------------------------------------------

vector_store = Chroma(
    collection_name="ai_assistant_documents",
    persist_directory="data/chroma_db",
    embedding_function=embedding_model,
)


# --------------------------------------------------
# 3. Check database
# --------------------------------------------------

count = vector_store._collection.count()

print("Documents in Chroma:", count)


# --------------------------------------------------
# 4. Create Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# --------------------------------------------------
# 5. User Question
# --------------------------------------------------

question = "What is Retrieval-Augmented Generation?"


# --------------------------------------------------
# 6. Retrieve relevant chunks
# --------------------------------------------------

results = retriever.invoke(question)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\nRetrieved documents:", len(results))


for i, document in enumerate(results, start=1):

    print(f"\n--- Retrieved Chunk {i} ---")

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)