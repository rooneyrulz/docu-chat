# Docu-Chat: AI-Powered Document Conversation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)

A conversational AI interface for your documents, powered by:
- Groq's ultra-fast LLMs (Mixtral/Llama2)
- FAISS vector similarity search
- LangChain RAG pipeline

## Features
📄 Chat with uploaded PDF documents  
⚡ Groq's high-speed inference  
🔍 Semantic search with FAISS vector store  
🛠️ Customizable chunking and embedding settings  

## Tech Stack
```python
streamlit==1.38.0
langchain-community==0.2.15
langchain-groq==0.1.9
faiss-cpu # Vector similarity search
unstructured[pdf]==0.15.0 # PDF processing