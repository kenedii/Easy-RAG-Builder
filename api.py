from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rag_utils
import llm_utils
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import warnings
import os

warnings.simplefilter("ignore", category=UserWarning)

# Global variables for shared encoders and retrieval cache
question_encoder = rag_utils.DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = rag_utils.DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = rag_utils.DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = rag_utils.DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
retrieval_cache: Dict[str, tuple] = {}  # collection_name -> (index, passages)
local_models: Dict[str, tuple] = {}  # Cache for loaded local models: model_name -> (generator_model, generator_tokenizer)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to get or create retrieval system for a collection
def get_retrieval_system(collection_name: str):
    """Load or create the retrieval system for a given collection."""
    if collection_name not in retrieval_cache:
        folder_path = os.path.join('data', collection_name)
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail="Collection not found")
        texts = rag_utils.extract_text_from_pdfs(folder_path)
        if not texts:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDFs in the collection")
        passages = rag_utils.split_into_passages(texts)
        index, passages = rag_utils.index_passages(passages, context_encoder, context_tokenizer)
        retrieval_cache[collection_name] = (index, passages)
    return retrieval_cache[collection_name]

# Request model with additional settings
class GenerateAnswerRequest(BaseModel):
    question: str
    model_name: str  # e.g., 'deepseek-chat', 'gpt-3.5-turbo-0125', 'gemini-2.0-flash'
    collection_name: str | None  # None if RAG is disabled
    max_tokens: int = 100
    temperature: float = 0.7
    num_passages: int = 5
    use_rag: bool = True  # New field to control RAG usage

# Endpoint for generating answers using a local model
@app.post("/generate_answer/local")
def generate_answer_local(request: GenerateAnswerRequest):
    question = request.question
    model_name = request.model_name
    collection_name = request.collection_name
    max_tokens = request.max_tokens
    temperature = request.temperature
    num_passages = request.num_passages
    use_rag = request.use_rag

    top_passages = []
    if use_rag:
        if not collection_name:
            raise HTTPException(status_code=400, detail="Collection name required when using RAG")
        index, passages = get_retrieval_system(collection_name)
        top_passages = rag_utils.retrieve_passages(
            question, index, passages, question_encoder, question_tokenizer, k=num_passages
        )

    if model_name not in local_models:
        try:
            generator_model, generator_tokenizer = local.load_model(model_name)
            local_models[model_name] = (generator_model, generator_tokenizer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load local model: {str(e)}")

    generator_model, generator_tokenizer = local_models[model_name]
    answer = local.generate_answer(question, top_passages, generator_model, generator_tokenizer)
    return {"answer": answer}

# Endpoint for generating answers using DeepSeek
@app.post("/generate_answer/deepseek")
def generate_answer_deepseek(request: GenerateAnswerRequest):
    question = request.question
    model_name = request.model_name
    collection_name = request.collection_name
    max_tokens = request.max_tokens
    temperature = request.temperature
    num_passages = request.num_passages
    use_rag = request.use_rag

    top_passages = []
    if use_rag:
        if not collection_name:
            raise HTTPException(status_code=400, detail="Collection name required when using RAG")
        index, passages = get_retrieval_system(collection_name)
        top_passages = rag_utils.retrieve_passages(
            question, index, passages, question_encoder, question_tokenizer, k=num_passages
        )

    answer = llm_utils.generate_answer(
        question, top_passages, model_name, api_type='deepseek', max_tokens=max_tokens, temperature=temperature
    )
    return {"answer": answer}

# Endpoint for generating answers using OpenAI
@app.post("/generate_answer/openai")
def generate_answer_openai(request: GenerateAnswerRequest):
    question = request.question
    model_name = request.model_name
    collection_name = request.collection_name
    max_tokens = request.max_tokens
    temperature = request.temperature
    num_passages = request.num_passages
    use_rag = request.use_rag

    top_passages = []
    if use_rag:
        if not collection_name:
            raise HTTPException(status_code=400, detail="Collection name required when using RAG")
        index, passages = get_retrieval_system(collection_name)
        top_passages = rag_utils.retrieve_passages(
            question, index, passages, question_encoder, question_tokenizer, k=num_passages
        )

    answer = llm_utils.generate_answer(
        question, top_passages, model_name, api_type='openai', max_tokens=max_tokens, temperature=temperature
    )
    return {"answer": answer}

# Endpoint for generating answers using Google Gemini
@app.post("/generate_answer/gemini")
def generate_answer_gemini(request: GenerateAnswerRequest):
    question = request.question
    model_name = request.model_name
    collection_name = request.collection_name
    max_tokens = request.max_tokens
    temperature = request.temperature
    num_passages = request.num_passages
    use_rag = request.use_rag

    top_passages = []
    if use_rag:
        if not collection_name:
            raise HTTPException(status_code=400, detail="Collection name required when using RAG")
        index, passages = get_retrieval_system(collection_name)
        top_passages = rag_utils.retrieve_passages(
            question, index, passages, question_encoder, question_tokenizer, k=num_passages
        )

    answer = llm_utils.generate_answer(
        question, top_passages, model_name, api_type='gemini', max_tokens=max_tokens, temperature=temperature
    )
    return {"answer": answer}

# Endpoint to clear cache for a collection
@app.post("/clear_cache")
def clear_cache(collection_name: str):
    if collection_name in retrieval_cache:
        del retrieval_cache[collection_name]
    return {"message": "Cache cleared"}

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "RAG API is running"}