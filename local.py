import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

def load_model(model_name, model_type):
    """Loads the specified local model and tokenizer based on model_type, and moves it to GPU if available."""
    if model_type == "causal":
        generator_model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_type == "seq2seq":
        generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        raise ValueError("Invalid model_type. Choose 'causal' or 'seq2seq'.")
    
    generator_tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator_model.to(device)
    
    print(f"Model loaded on device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    return generator_model, generator_tokenizer

def generate_answer(question, top_passages, generator_model, generator_tokenizer, model_type):
    """Generates an answer using the local model based on model_type."""
    if model_type == "causal":
        # For causal models like bloom-1b or distilgpt2
        prompt = f"Question: {question}\nContext: {' '.join(top_passages)}\nAnswer:"
        inputs = generator_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {key: value.to(generator_model.device) for key, value in inputs.items()}
        outputs = generator_model.generate(**inputs, max_new_tokens=200, num_beams=5, early_stopping=True)
        answer_ids = outputs[0][inputs['input_ids'].shape[1]:]
        answer = generator_tokenizer.decode(answer_ids, skip_special_tokens=True)
    elif model_type == "seq2seq":
        # For seq2seq models like t5-small
        context = ' '.join(top_passages)
        input_text = f"question: {question} context: {context}"
        inputs = generator_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {key: value.to(generator_model.device) for key, value in inputs.items()}
        outputs = generator_model.generate(**inputs, max_new_tokens=200, num_beams=5, early_stopping=True)
        answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        raise ValueError("Invalid model_type. Choose 'causal' or 'seq2seq'.")
    
    return answer