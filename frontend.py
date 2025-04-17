import streamlit as st
import os
import requests
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
API_URL = "http://localhost:8000"  # Adjust if your FastAPI server runs on a different port

# Helper functions for data management
def list_collections():
    """List all data collections in the 'data' folder."""
    if not DATA_DIR.exists():
        return []
    return [d.name for d in DATA_DIR.iterdir() if d.is_dir()]

def create_collection(name):
    """Create a new collection folder inside 'data'."""
    collection_path = DATA_DIR / name
    if not collection_path.exists():
        collection_path.mkdir(parents=True)

def rename_collection(old_name, new_name):
    """Rename an existing collection folder."""
    old_path = DATA_DIR / old_name
    new_path = DATA_DIR / new_name
    if old_path.exists() and not new_path.exists():
        old_path.rename(new_path)

def list_files(collection_name):
    """List all files in a collection."""
    collection_path = DATA_DIR / collection_name
    if collection_path.exists():
        return [f.name for f in collection_path.iterdir() if f.is_file()]
    return []

def delete_file(collection_name, filename):
    """Delete a file from a collection."""
    file_path = DATA_DIR / collection_name / filename
    if file_path.exists():
        file_path.unlink()

def upload_files(collection_name, uploaded_files):
    """Upload files to a collection."""
    collection_path = DATA_DIR / collection_name
    if not collection_path.exists():
        collection_path.mkdir(parents=True)
    for uploaded_file in uploaded_files:
        with open(collection_path / uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

# Streamlit app
st.title("RAG System Frontend")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Chat", "Data"])

# Chat Page
if page == "Chat":
    st.header("Chat with RAG System")

    # LLM selection
    llm_option = st.selectbox("Select LLM", ["DeepSeek", "OpenAI"])

    # Data collection selection
    collections = list_collections()
    if collections:
        selected_collection = st.selectbox("Select Data Collection", collections)
    else:
        st.warning("No data collections available. Please create one in the Data page.")
        selected_collection = None

    # Chat settings
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=100)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7)

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if selected_collection:
            # Determine API endpoint and model based on LLM selection
            if llm_option == "DeepSeek":
                endpoint = "/generate_answer/deepseek"
                model_name = "deepseek-chat"  # Default DeepSeek model
            else:
                endpoint = "/generate_answer/openai"
                model_name = "gpt-3.5-turbo"  # Default OpenAI model

            # Prepare request data
            request_data = {
                "question": prompt,
                "model_name": model_name,
                "collection_name": selected_collection,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Send request to API
            with st.spinner("Generating answer..."):
                try:
                    response = requests.post(f"{API_URL}{endpoint}", json=request_data)
                    response.raise_for_status()
                    answer = response.json()["answer"]
                except Exception as e:
                    answer = f"Error: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            st.error("Please select a data collection.")

# Data Page
elif page == "Data":
    st.header("Data Management")

    # View all collections
    st.subheader("Data Collections")
    collections = list_collections()
    if collections:
        selected_collection = st.selectbox("Select Collection", collections, key="data_collection_select")
    else:
        selected_collection = None
        st.write("No collections available.")

    # Create new collection
    with st.expander("Create New Collection"):
        new_collection_name = st.text_input("Collection Name")
        if st.button("Create"):
            if new_collection_name:
                create_collection(new_collection_name)
                st.success(f"Collection '{new_collection_name}' created.")
                st.rerun()  # Refresh the page to update the collection list
            else:
                st.error("Please enter a collection name.")

    if selected_collection:
        # View and delete files
        st.subheader(f"Files in '{selected_collection}'")
        files = list_files(selected_collection)
        if files:
            for file in files:
                col1, col2 = st.columns([3, 1])
                col1.write(file)
                if col2.button("Delete", key=f"delete_{file}"):
                    delete_file(selected_collection, file)
                    # Clear API cache for this collection
                    requests.post(f"{API_URL}/clear_cache", json={"collection_name": selected_collection})
                    st.success(f"File '{file}' deleted.")
                    st.rerun()
        else:
            st.write("No files in this collection.")

        # Upload files (drag and drop)
        with st.expander("Upload Files"):
            uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf"])
            if st.button("Upload"):
                if uploaded_files:
                    upload_files(selected_collection, uploaded_files)
                    # Clear API cache for this collection
                    requests.post(f"{API_URL}/clear_cache", json={"collection_name": selected_collection})
                    st.success("Files uploaded successfully.")
                    st.rerun()
                else:
                    st.error("Please select files to upload.")

        # Rename collection
        with st.expander("Rename Collection"):
            new_name = st.text_input("New Collection Name")
            if st.button("Rename"):
                if new_name:
                    rename_collection(selected_collection, new_name)
                    st.success(f"Collection renamed to '{new_name}'.")
                    st.rerun()
                else:
                    st.error("Please enter a new name.")