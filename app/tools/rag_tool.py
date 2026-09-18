from langchain_core.tools import tool

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Create Embedding Model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Load Chroma Vector Database
# --------------------------------------------------

vector_store = Chroma(
    collection_name="ai_assistant_documents",
    persist_directory="data/chroma_db",
    embedding_function=embedding_model,
)


# --------------------------------------------------
# 3. Create Retriever
# --------------------------------------------------

# retriever = vector_store.as_retriever(
#     search_kwargs={
#         "k": 2
#     }
# )

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# --------------------------------------------------
# 4. Create RAG Search Tool
# --------------------------------------------------

@tool
def rag_search(question: str) -> str:
    """
    Search the knowledge base for information
    relevant to the user's question.

    Use this tool when the user asks a question
    that may be answered using the provided documents.
    """

    documents = retriever.invoke(question)

    if not documents:

        return "No relevant information was found."

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context