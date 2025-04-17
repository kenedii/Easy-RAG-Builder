from openai import OpenAI
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables from the .env file
load_dotenv('do_not_commit.env')

# Load API keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize OpenAI client (assumes OPENAI_API_KEY is set in the environment)
openai_client_instance = OpenAI()

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# DeepSeek-specific configuration
DEEPSEEK_MODEL_NAME = "deepseek-chat"
DEEPSEEK_API_URL = "https://api.deepseek.com"

def _send_deepseek_message(messages, max_tokens=100, temperature=0.7):
    """Send messages to the DeepSeek API using OpenAI SDK and return a standardized dictionary."""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not set in environment variables")
    
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        assistant_message = {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content
        }
        return {"choices": [{"message": assistant_message}]}
    except Exception as err:
        print(f"[ERROR] DeepSeek API error: {err}")
        return None

def generate_answer(question, top_passages, model_name, api_type='openai', max_tokens=100, temperature=0.7):
    """
    Generates an answer using the specified API (OpenAI, DeepSeek, or Gemini) with the given model.
    
    Args:
        question (str): The input question.
        top_passages (list): List of retrieved passages (empty if RAG is disabled).
        model_name (str): The model to use (e.g., 'gpt-3.5-turbo' for OpenAI, 'deepseek-chat' for DeepSeek, 'gemini-pro' for Gemini).
        api_type (str): 'openai', 'deepseek', or 'gemini' to specify the API provider.
        max_tokens (int): Maximum number of tokens to generate (default: 100).
        temperature (float): Sampling temperature (default: 0.7).
    
    Returns:
        str: Generated answer.
    """
    # Construct prompt based on whether passages are provided
    if top_passages:
        prompt = (
            f"Question: {question}\n"
            f"Context: Below are relevant excerpts from the documentation of the ECGeniuses machine:\n"
            f"{' '.join([f'- {passage}' for passage in top_passages])}\n"
            f"Answer: Provide a detailed, step-by-step response using insights on the context. If you do not have enough information to give an accurate response, let the user know."
        )
    else:
        prompt = (
            f"Question: {question}\n"
            f"Answer: Provide a detailed response to the question. If you do not have enough information to give an accurate response, let the user know."
        )
    
    messages = [{"role": "user", "content": prompt}]

    if api_type == 'openai':
        response = openai_client_instance.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = response.choices[0].message.content.strip()
    elif api_type == 'deepseek':
        response = _send_deepseek_message(messages, max_tokens=max_tokens, temperature=temperature)
        if response is None:
            return "Error: Could not generate an answer using DeepSeek API."
        answer = response["choices"][0]["message"]["content"].strip()
    elif api_type == 'gemini':
        try:
            model = genai.GenerativeModel(model_name)
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            answer = response.text.strip()
        except Exception as e:
            print(f"[ERROR] Gemini API error: {e}")
            return "Error: Could not generate an answer using Gemini API."
    else:
        raise ValueError("Invalid api_type. Choose 'openai', 'deepseek', or 'gemini'.")

    return answer