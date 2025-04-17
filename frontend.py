import streamlit as st
import os
import requests
from pathlib import Path
import shutil
import json
import uuid
from datetime import datetime
import re

# Configuration
DATA_DIR = Path("data")
CHAT_HISTORY_DIR = Path("chat_history")
API_URL = "http://localhost:8000"  # Adjust if your FastAPI server runs on a different port

# Ensure chat_history directory exists
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

# Model options for each provider
MODEL_OPTIONS = {
    "DeepSeek": [
        {"name": "deepseek-chat", "display": "DeepSeek Chat (Default)"},
        {"name": "deepseek-reasoner", "display": "DeepSeek Reasoner"}
    ],
    "OpenAI": [
        {"name": "gpt-3.5-turbo-0125", "display": "GPT 3.5 (Default)"},
        {"name": "gpt-4o-mini-2024-07-18", "display": "GPT 4o"},
        {"name": "o1-2024-12-17", "display": "GPT o1"},
        {"name": "o3-mini-2025-01-31", "display": "GPT o3"},
        {"name": "gpt-4.1-2025-04-14", "display": "GPT 4.1"}
    ],
    "Google": [
        {"name": "gemini-2.0-flash", "display": "Gemini 2.0 Flash (Default)"},
        {"name": "gemini-1.5-flash", "display": "Gemini 1.5 Flash"},
        {"name": "gemini-1.5-pro", "display": "Gemini 1.5 Pro"},
        {"name": "gemini-2.5-pro-preview-03-25", "display": "Gemini 2.5 Pro Preview"}
    ]
}

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

def delete_collection(collection_name):
    """Delete a collection folder and all its contents."""
    collection_path = DATA_DIR / collection_name
    if collection_path.exists():
        shutil.rmtree(collection_path)
        try:
            requests.post(f"{API_URL}/clear_cache", json={"collection_name": collection_name})
        except Exception as e:
            st.warning(f"Failed to clear API cache: {str(e)}")

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
        requests.post(f"{API_URL}/clear_cache", json={"collection_name": collection_name})

def upload_files(collection_name, uploaded_files):
    """Upload files to a collection."""
    collection_path = DATA_DIR / collection_name
    if not collection_path.exists():
        collection_path.mkdir(parents=True)
    for uploaded_file in uploaded_files:
        with open(collection_path / uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
    requests.post(f"{API_URL}/clear_cache", json={"collection_name": collection_name})

# Helper functions for chat history management
def list_chat_histories():
    """List all chat history files with metadata."""
    chat_histories = []
    for file_path in CHAT_HISTORY_DIR.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            preview = get_chat_preview(data.get("messages", []))
            chat_histories.append({
                "id": file_path.stem,
                "name": data.get("name", ""),
                "preview": preview,
                "timestamp": data.get("timestamp", ""),
                "file_path": file_path
            })
        except Exception as e:
            st.warning(f"Failed to load chat {file_path}: {str(e)}")
    # Sort by timestamp (newest first)
    return sorted(chat_histories, key=lambda x: x["timestamp"], reverse=True)

def get_chat_preview(messages):
    """Get a preview of the first few words from the first user message."""
    for msg in messages:
        if msg["role"] == "user":
            words = re.findall(r'\w+', msg["content"])
            return ' '.join(words[:5]) + ("..." if len(words) > 5 else "")
    return "No messages"

def save_chat_history(chat_id, messages, name=""):
    """Save chat history to a JSON file."""
    chat_data = {
        "id": chat_id,
        "name": name,
        "timestamp": datetime.utcnow().isoformat(),
        "messages": messages
    }
    file_path = CHAT_HISTORY_DIR / f"{chat_id}.json"
    with open(file_path, 'w') as f:
        json.dump(chat_data, f, indent=2)

def load_chat_history(chat_id):
    """Load chat history from a JSON file."""
    file_path = CHAT_HISTORY_DIR / f"{chat_id}.json"
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get("messages", []), data.get("name", "")
    return [], ""

def rename_chat_history(chat_id, new_name):
    """Rename a chat history by updating its JSON file."""
    file_path = CHAT_HISTORY_DIR / f"{chat_id}.json"
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
        data["name"] = new_name
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

def delete_chat_history(chat_id):
    """Delete a chat history file."""
    file_path = CHAT_HISTORY_DIR / f"{chat_id}.json"
    if file_path.exists():
        file_path.unlink()

# Streamlit app
st.title("RAG System Frontend")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Chat", "Data"])

# Chat selection in sidebar for "Chat" page
if page == "Chat":
    st.sidebar.subheader("Chat Histories")
    chat_histories = list_chat_histories()
    chat_options = ["New Chat"] + [f"{h['name'] or 'Chat ' + h['id'][:8]}: {h['preview']}" for h in chat_histories]
    selected_chat = st.sidebar.selectbox("Select Chat", chat_options, key="chat_select")

    if selected_chat == "New Chat":
        # Start a new chat
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.chat_name = ""
    else:
        # Load selected chat
        selected_chat_id = chat_histories[chat_options.index(selected_chat) - 1]["id"]
        if selected_chat_id != st.session_state.chat_id:
            st.session_state.chat_id = selected_chat_id
            st.session_state.messages, st.session_state.chat_name = load_chat_history(selected_chat_id)

# Main area for "Chat" page
if page == "Chat":
    st.header("Chat with RAG System")

    # Display current chat name
    if st.session_state.chat_name:
        st.subheader(f"Chat: {st.session_state.chat_name}")
    else:
        st.subheader("New Chat")

    # Rename Chat
    with st.expander("Rename Chat"):
        new_chat_name = st.text_input("Chat Name", value=st.session_state.chat_name)
        if st.button("Rename"):
            if new_chat_name:  # Allow empty names to reset to default
                st.session_state.chat_name = new_chat_name
                save_chat_history(st.session_state.chat_id, st.session_state.messages, st.session_state.chat_name)
                st.success("Chat renamed successfully.")
                st.rerun()
            else:
                st.session_state.chat_name = ""
                save_chat_history(st.session_state.chat_id, st.session_state.messages, "")
                st.success("Chat name reset.")
                st.rerun()

    # Delete Chat
    if selected_chat != "New Chat":
        with st.expander("Delete Chat"):
            st.warning("This will permanently delete the chat history.")
            if st.button("Delete Chat"):
                delete_chat_history(st.session_state.chat_id)
                st.session_state.chat_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.chat_name = ""
                st.success("Chat deleted successfully.")
                st.rerun()

    # LLM provider selection
    provider = st.selectbox("Select LLM Provider", ["DeepSeek", "OpenAI", "Google"])

    # Model selection based on provider
    model_options = MODEL_OPTIONS[provider]
    model_display_names = [model["display"] for model in model_options]
    selected_model_display = st.selectbox("Select Model", model_display_names)
    selected_model = next(model["name"] for model in model_options if model["display"] == selected_model_display)

    # Data collection selection
    collections = list_collections()
    if collections:
        selected_collection = st.selectbox("Select Data Collection", collections)
    else:
        st.warning("No data collections available. Please create one in the Data page.")
        selected_collection = None

    # Chat settings
    st.subheader("Chat Settings")
    max_tokens = st.slider("Max Tokens (Output)", min_value=50, max_value=15000, value=3000, help="Controls the maximum number of tokens in the generated response.")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, help="Controls the randomness of the response.")
    num_passages = st.slider("Number of Context Passages", min_value=1, max_value=10, value=5, help="Controls how many retrieved passages are included as context in the prompt.")

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if selected_collection:
            # Determine API endpoint based on provider
            if provider == "DeepSeek":
                endpoint = "/generate_answer/deepseek"
            elif provider == "OpenAI":
                endpoint = "/generate_answer/openai"
            elif provider == "Google":
                endpoint = "/generate_answer/gemini"

            # Prepare request data
            request_data = {
                "question": prompt,
                "model_name": selected_model,
                "collection_name": selected_collection,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "num_passages": num_passages,
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

            # Save chat history
            save_chat_history(st.session_state.chat_id, st.session_state.messages, st.session_state.chat_name)
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
                st.rerun()
            else:
                st.error("Please enter a collection name.")

    if selected_collection:
        # Delete collection
        with st.expander("Delete Collection"):
            st.warning("This will permanently delete the collection and all its files.")
            if st.button(f"Delete '{selected_collection}'"):
                delete_collection(selected_collection)
                st.success(f"Collection '{selected_collection}' deleted.")
                st.rerun()

        # View and delete files
        st.subheader(f"Files in '{selected_collection}'")
        files = list_files(selected_collection)
        if files:
            for file in files:
                col1, col2 = st.columns([3, 1])
                col1.write(file)
                if col2.button("Delete", key=f"delete_{file}"):
                    delete_file(selected_collection, file)
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