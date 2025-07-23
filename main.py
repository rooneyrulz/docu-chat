import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

import page_config
from utils import (load_pdf, save_uploaded_file,
                   split_documents, create_vector_store, create_conversational_chain)
from config import (DOCUMENTS_DIR_PATH, ENV_FILE_PATH, FAISS_INDEX_PATH, SESSION_STATES)

page_config.set_page_config()

# Directory paths
CWD = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(CWD, DOCUMENTS_DIR_PATH)
FAISS_INDEX_DIR = os.path.join(CWD, FAISS_INDEX_PATH)

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Load environment variables
env_path = os.path.join(CWD, ENV_FILE_PATH)
load_dotenv(dotenv_path=env_path)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

print(f"GROQ API KEY ---> {os.getenv("GROQ_API_KEY")}")

# Initialize the session states
for key, default_value in SESSION_STATES.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def load_document(dir_path: str, file_name: str) -> List:
    """Load the documents from the directory."""
    documents = load_pdf(dir_path, file_name)
    chunks = split_documents(documents)
    return documents, chunks


def load_vector_store(document_chunks: List) -> FAISS:
    """Load and create the vector store."""
    vector_store = create_vector_store(document_chunks)
    return vector_store


# Initialize conversational chain if vector store is available
if st.session_state.vector_store and not st.session_state.conversation_chain:
    st.session_state.conversation_chain = create_conversational_chain(
        st.session_state.vector_store)

uploaded_file = st.file_uploader(
    "Upload your PDF file", type="pdf", disabled=st.session_state.is_vector_store_creating)
if uploaded_file:
    try:
        file_name = save_uploaded_file(uploaded_file, DOCUMENTS_DIR)
        st.success(f"File '{file_name}' uploaded successfully!", icon="✅")
        st.session_state.uploaded_file = file_name
    except Exception as e:
        st.error(f"Error uploading document: {e}")

if st.button("Update Knowledge Base 🗎", key="update_knowledge_base", type="primary", use_container_width=True):
    if st.session_state.uploaded_file is not None:
        st.session_state.is_vector_store_creating = True
        with st.spinner("Hold on, Updating the knowledge base for you..."):
            try:
                documents, chunks = load_document(
                    DOCUMENTS_DIR, st.session_state.uploaded_file)
                vector_store = load_vector_store(chunks)
                st.session_state.vector_store = vector_store
                st.session_state.conversation_chain = create_conversational_chain(vector_store)
                st.session_state.documents = [
                    doc.metadata['source'] for doc in documents]
                st.success("Knowledge base updated! Let's start chatting :)", icon='🎉')
            except Exception as e:
                st.error(f"Error updating knowledge base: {e}")
            finally:
                st.session_state.is_vector_store_creating = False
    else:
        st.warning("Please upload the document first!")

# Load and show already vectorized documents if Chroma store exists
if st.session_state.vector_store and st.session_state.documents:
    st.markdown("<h5>Loaded documents for Q&A:</h5>", unsafe_allow_html=True)
    for doc_name in st.session_state.documents:
        st.write(f"- {doc_name}")

# Show workflow only if the vector store is not ready and the user hasn't started chatting
# if not st.session_state.get('vector_store') or not st.session_state.get('conversation_chain'):
st.markdown("""
    <div style="color: #999; font-style: italic; margin-top: 1rem;">
        <strong>Workflow:</strong>
        <ol>
            <li>Upload your document</li>
            <li>Update the knowledge base</li>
            <li>Start chatting</li>
        </ol>
    </div>
""", unsafe_allow_html=True)


# Display the previous history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input(
    "Let's chat with your document...", disabled=st.session_state.is_vector_store_creating)

if user_input:
    # Add the user's question to the chat history
    st.session_state.chat_history.append(
        {"role": "human", "content": user_input})

    with st.chat_message("human"):
        st.markdown(user_input)

    if st.session_state.conversation_chain:
        with st.chat_message("ai"):
            thinking_message = st.markdown("Thinking...")
            response = st.session_state.conversation_chain(
                {"question": user_input})
            assistant_response = response["answer"]
            thinking_message.empty()
            st.markdown(assistant_response)
            st.session_state.chat_history.append(
                {"role": "ai", "content": assistant_response})
    else:
        st.warning("Please update the knowledge base first!.")
