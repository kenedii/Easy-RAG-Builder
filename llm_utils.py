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

def generate_answer(messages, top_passages, model_name, api_type='openai', max_tokens=100, temperature=0.7, system_prompt=""):
    """
    Generates an answer using the specified API with the given message history, passages, and system prompt.

    Args:
        messages (list): List of Message objects with role and content.
        top_passages (list): List of dicts with 'text', 'file_name', and 'page_number'.
        model_name (str): The model to use.
        api_type (str): 'openai', 'deepseek', or 'gemini'.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        system_prompt (str): Custom system prompt to define assistant behavior.

    Returns:
        str: Generated answer with citations.
    """
    # Construct prompt with system prompt and context
    prompt = f"{system_prompt}\n\n" if system_prompt else ""

    if top_passages:
        context = "\n".join([
            f"[{i+1}] {passage['text']} (Source: {passage['file_name']}, Page {passage['page_number']})"
            for i, passage in enumerate(top_passages)
        ])
        prompt += (
            f"Use the following context to answer the question, citing sources with [number] in your response. "
            f"Include a 'References' section at the end listing the sources used.\n\n"
            f"Context:\n{context}\n\n"
            f"Conversation History:\n"
        )
        for msg in messages:
            prompt += f"{msg.role.capitalize()}: {msg.content}\n"
        prompt += (
            f"\nAnswer: Provide a detailed response, citing relevant sources with [number]. "
            f"If you lack information, state so. End with a 'References' section listing the cited sources."
        )
    else:
        prompt += (
            f"Answer based on your knowledge, using the conversation history below.\n\n"
            f"Conversation History:\n"
        )
        for msg in messages:
            prompt += f"{msg.role.capitalize()}: {msg.content}\n"
        prompt += f"\nAnswer: Provide a detailed response. If you lack information, state so."

    api_messages = [{"role": "user", "content": prompt}]

    if api_type == 'openai':
        response = openai_client_instance.chat.completions.create(
            model=model_name,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = response.choices[0].message.content.strip()
    elif api_type == 'deepseek':
        response = _send_deepseek_message(api_messages, max_tokens=max_tokens, temperature=temperature)
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