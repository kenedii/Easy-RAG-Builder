# Easy-RAG-Builder

This is an interface to build RAG models and chat with different LLMs. 

You can build a new RAG by creating a 'Data Collection' and adding your data (only PDF supported).

Then, go to the chat and talk to your Data Collection. 

You can also save chats, disable RAG functionality and talk to the LLM normally, and disable including the conversation history in your chats.


The LLMs it supports are:
- DeepSeek API chat/reasoning
- OpenAI API 3.5, 4o, o1, o3, 4.1
- Gemini API 2.0-flash, 1.5-flash, 1.5-pro, 2.5-pro-preview-03-25
- Local Models: t5-small (Seq2Seq), bigscience/bloom-1b1 (Casual), distilgpt2 (Casual)

Dependencies:
- Python (3.11.5 used)
- Pip (23.2.1 used)

How to run: - Add API keys to `do_not_commit.env`

Windows:
- Double click launch_frontend.bat

Other:
- pip install -r requirements.txt
- uvicorn api:app --port=8000
- streamlit run frontend.py

if api has errors run `set KMP_DUPLICATE_LIB_OK=TRUE` before launching api

![YNFsDzj](https://github.com/user-attachments/assets/626390e1-0d13-4cbe-b93c-43617b263ad1)
