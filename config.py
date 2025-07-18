from langchain_community.vectorstores import FAISS
# Constants
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3

CHAIN_TYPE = "stuff"
CHAIN_VERBOSE = True

# Relative paths to main.py file
DOCUMENTS_DIR_PATH = "documents"
ENV_FILE_PATH = ".env"
FAISS_INDEX_PATH = "faiss_index"

PAGE_TITLE = "Document Q&A"
PAGE_ICON = "🗎"

APP_HEADING = "Document Q&A"
APP_DESCRIPTION = "Chat with your document!"
APP_ICON = "🗎"

# Avoid circular imports
def init_vector_store() -> FAISS:
    from utils import load_faiss_index, embedding

    store = load_faiss_index(FAISS_INDEX_PATH, embedding)
    return store

# session states with default values
SESSION_STATES = {"chat_history": [],
                  "documents": [],
                  "vector_store": init_vector_store(),
                  "conversation_chain": None,
                  "is_vector_store_creating": False,
                  "uploaded_file": None}
