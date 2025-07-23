import os
from typing import List, Optional

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config import (CHAT_MODEL, TEMPERATURE,
                    CHAIN_TYPE, CHAIN_VERBOSE, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, FAISS_INDEX_PATH)


def save_uploaded_file(uploaded_file, documents_directory: str) -> str:
    """Save the uploaded file to the documents directory.

    Args:
        uploaded_file: The file to be saved.
        documents_directory (str): The directory where the file will be saved.

    Returns:
        str: The name of the saved file.
    """
    file_path = os.path.join(documents_directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name


def load_pdf(directory: str, filename: str) -> List:
    """Load documents from a specified directory.

    Args:
        directory (str): The directory containing the PDF file.
        filename (str): The name of the PDF file.

    Returns:
        List: A list of loaded documents.
    """
    file_path = os.path.join(directory, filename)
    loader = UnstructuredPDFLoader(file_path)
    return loader.load()


def split_documents(documents: List, chunk_size: Optional[int] = CHUNK_SIZE, chunk_overlap: Optional[int] = CHUNK_OVERLAP) -> List:
    """Split documents into smaller chunks.

    Args:
        documents (List): The documents to be split.
        chunk_size (Optional[int]): The size of each chunk. Defaults to CHUNK_SIZE.
        chunk_overlap (Optional[int]): The overlap between chunks. Defaults to CHUNK_OVERLAP.

    Returns:
        List: A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def load_embedding_model(model: str) -> HuggingFaceEmbeddings:
    """Load the embedding model.

    Args:
        model (str): The name of the model to load.

    Returns:
        HuggingFaceEmbeddings: The loaded embedding model.
    """
    return HuggingFaceEmbeddings(model_name=model)

def load_faiss_index(index: str, embedding: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.load_local(index, embedding, allow_dangerous_deserialization=True)


embedding = load_embedding_model(model=EMBEDDING_MODEL)
def create_vector_store(texts: List) -> FAISS:
    """Create a vector store from the given texts.

    Args:
        texts (List): The texts to create the vector store from.

    Returns:
        FAISS: The created vector store.
    """
    index_name = FAISS_INDEX_PATH
    index = FAISS.from_documents(
        documents=texts,
        embedding=embedding,
    )
    index.save_local(index_name)
    return load_faiss_index(index_name, embedding)


def create_conversational_chain(
    v_store: FAISS,
    model: Optional[str] = CHAT_MODEL,
    temperature: Optional[float] = TEMPERATURE,
    chain_type: Optional[str] = CHAIN_TYPE,
    verbose: Optional[bool] = CHAIN_VERBOSE
) -> ConversationalRetrievalChain:
    """Create a conversational retrieval chain.

    Args:
        v_store (FAISS): The vector store to use for retrieval.
        model (Optional[str]): The model to use for the language model. Defaults to CHAT_MODEL.
        temperature (Optional[float]): The temperature for the language model. Defaults to TEMPERATURE.
        chain_type (Optional[str]): The type of chain to create. Defaults to CHAIN_TYPE.
        verbose (Optional[bool]): Whether to enable verbose output. Defaults to CHAIN_VERBOSE.

    Returns:
        ConversationalRetrievalChain: The created conversational retrieval chain.

    Raises:
        ValueError: If the vector store is not initialized properly.
    """
    if v_store is None:
        raise ValueError(
            "The vector store must be initialized before creating the chain.")

    retriever = v_store.as_retriever()
    llm = ChatGroq(model=model, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type,
        memory=memory,
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=verbose,
    )
