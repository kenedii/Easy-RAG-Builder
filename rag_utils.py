import os
import numpy as np
import pdfplumber
import faiss
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

def extract_text_from_pdfs(folder_path):
    """Extracts text from all PDFs in the given folder."""
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        texts.append(text)
    return texts

def split_into_passages(texts, max_length=200):
    """Splits text into smaller passages for retrieval."""
    passages = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), max_length):
            passage = ' '.join(words[i:i + max_length])
            passages.append(passage)
    return passages

def index_passages(passages, context_encoder, context_tokenizer):
    """Encodes passages into vectors and creates a FAISS index."""
    passage_vectors = []
    for passage in passages:
        inputs = context_tokenizer(passage, return_tensors='pt', truncation=True, max_length=512)
        vector = context_encoder(**inputs).pooler_output.detach().numpy()
        passage_vectors.append(vector)
    passage_vectors = np.vstack(passage_vectors)
    index = faiss.IndexFlatIP(768)  # DPR vector dimension is 768
    index.add(passage_vectors)
    return index, passages

def retrieve_passages(question, index, passages, question_encoder, question_tokenizer, k=5):
    """Retrieves the top-k most relevant passages for a given question."""
    question_inputs = question_tokenizer(question, return_tensors='pt', truncation=True, max_length=512)
    question_vector = question_encoder(**question_inputs).pooler_output.detach().numpy()
    distances, indices = index.search(question_vector, k)
    top_passages = [passages[idx] for idx in indices[0]]
    return top_passages

def setup_retrieval(folder_path):
    """Sets up the retrieval system with PDFs from the specified folder."""
    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    
    texts = extract_text_from_pdfs(folder_path)
    if not texts:
        raise ValueError("No text could be extracted from the PDFs.")
    passages = split_into_passages(texts)
    index, passages = index_passages(passages, context_encoder, context_tokenizer)
    
    return index, passages, question_encoder, question_tokenizer