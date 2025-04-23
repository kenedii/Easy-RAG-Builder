from unstructured.partition.auto import partition
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import faiss
import numpy as np
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import warnings

warnings.filterwarnings("ignore")

def extract_text_from_documents(folder_path: str) -> List[Dict]:
    """
    Extract text from all supported document files in the specified folder, including file name and page number if available.

    Args:
        folder_path (str): Path to the folder containing document files.

    Returns:
        List[Dict]: List of dictionaries with 'text', 'file_name', and 'page_number'.
    """
    texts = []
    folder = Path(folder_path)
    for file in folder.iterdir():
        if file.is_file():
            try:
                # Use Unstructured.io to partition the file into elements
                elements = partition(filename=str(file))
                for element in elements:
                    text = element.text.strip()
                    if text:
                        # Extract page number from metadata if available; otherwise, set to None
                        page_number = element.metadata.page_number if hasattr(element.metadata, 'page_number') else None
                        texts.append({
                            "text": text,
                            "file_name": file.name,
                            "page_number": page_number
                        })
            except Exception as e:
                print(f"[ERROR] Failed to process {file}: {e}")
    return texts

def split_into_passages(texts: List[Dict], max_length: int = 200) -> List[Dict]:
    """
    Split extracted texts into passages, preserving metadata.

    Args:
        texts (List[Dict]): List of text dicts with 'text', 'file_name', and 'page_number'.
        max_length (int): Maximum number of words per passage.

    Returns:
        List[Dict]: List of passage dicts with 'text', 'file_name', and 'page_number'.
    """
    passages = []
    for item in texts:
        text = item["text"]
        file_name = item["file_name"]
        page_number = item["page_number"]
        words = text.split()
        if len(words) <= max_length:
            passages.append(item)
        else:
            for i in range(0, len(words), max_length):
                passage_text = " ".join(words[i:i + max_length])
                passages.append({
                    "text": passage_text,
                    "file_name": file_name,
                    "page_number": page_number
                })
    return passages

def index_passages(passages: List[Dict], context_encoder: DPRContextEncoder, context_tokenizer: DPRContextEncoderTokenizer) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """
    Index passages using DPR context encoder, preserving metadata.

    Args:
        passages (List[Dict]): List of passage dicts with 'text', 'file_name', and 'page_number'.
        context_encoder (DPRContextEncoder): DPR context encoder model.
        context_tokenizer (DPRContextEncoderTokenizer): DPR context tokenizer.

    Returns:
        Tuple[faiss.IndexFlatL2, List[Dict]]: FAISS index and passages with metadata.
    """
    embeddings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_encoder = context_encoder.to(device)
    context_encoder.eval()

    for passage in passages:
        inputs = context_tokenizer(
            passage["text"],
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embedding = context_encoder(**inputs).pooler_output
        embeddings.append(embedding.cpu().numpy())

    embeddings = np.vstack(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, passages

def retrieve_passages(
    question: str,
    index: faiss.IndexFlatL2,
    passages: List[Dict],
    question_encoder: DPRQuestionEncoder,
    question_tokenizer: DPRQuestionEncoderTokenizer,
    k: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Retrieve top-k relevant passages for the given question, including metadata.

    Args:
        question (str): The input question.
        index (faiss.IndexFlatL2): FAISS index of passage embeddings.
        passages (List[Dict]): List of passage dicts with 'text', 'file_name', and 'page_number'.
        question_encoder (DPRQuestionEncoder): DPR question encoder model.
        question_tokenizer (DPRQuestionEncoderTokenizer): DPR question tokenizer.
        k (int): Number of passages to retrieve.

    Returns:
        Tuple[List[Dict], List[Dict]]: List of top passages and their metadata (file_name, page_number).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_encoder = question_encoder.to(device)
    question_encoder.eval()

    inputs = question_tokenizer(
        question,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        question_embedding = question_encoder(**inputs).pooler_output.cpu().numpy()

    distances, indices = index.search(question_embedding, k)
    top_passages = [passages[i] for i in indices[0]]
    # Include page_number in sources only if it exists
    sources = []
    for p in top_passages:
        source = {"file_name": p["file_name"]}
        if p["page_number"] is not None:
            source["page_number"] = p["page_number"]
        sources.append(source)
    return top_passages, sources