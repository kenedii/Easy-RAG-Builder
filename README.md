# Easy-RAG-Builder

How to run:
- pip install -r requirements.txt
- Add API keys to `do_not_commit.env`
- uvicorn api:app --port=8000
- streamlit run frontend.py

if api has errors run `set KMP_DUPLICATE_LIB_OK=TRUE` before launching api
