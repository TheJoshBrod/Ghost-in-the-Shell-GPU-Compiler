import regex as re
import ollama as ol
import google.generativeai as genai
from src.prompts import prompts


def clean_output(raw: str) -> str:
    """Removes ```CUDA, ```, and other markdown fluff."""

    # Remove the suffix first
    if raw.endswith("```"):
        raw = raw[:-3]  # remove last 3 chars
    
    # Remove the prefix like ```cuda
    raw = re.sub(r"^```(\w+)?\n", "", raw)

    return raw

def ollama_generator(msg: str, model: str = "llama3.2:latest", outputIR: str = "CUDA"):
    """Initial generation of kernel/IR

    Args:
        msg (str): Context for LLM to generate Kernel/IR
        model (str, optional): Which Ollama model to use for LLM. Defaults to "llama3.2:latest".
        outputIR (str, optional): What is the desired output IR type. Defaults to "CUDA".

    Returns:
        str: Cleaned up version of the generated Kernel/IR 
    """
    print("Generating code...")
    sys_prompt = prompts.get_generation_sys_prompt(outputIR)
    response = ol.chat(model=model, messages=[{"role": "system", "content": sys_prompt},{"role": "user", "content": msg}])
    raw_output = response['message']['content']
    cleaned_output = clean_output(raw_output)
    print("Code generated...")
    return cleaned_output

def ollama_fixer(kernel: str, error: str, msg: str, model: str = "llama3.2:latest", outputIR: str = "CUDA"):
    """Fixes the previously generated kernel 

    Args:
        kernel (str): Previous version of the malformed/incorrect Kernel generated
        error (str): Custom error message to inform what the LLM did wrong 
        msg (str): Context the ORIGINAL LLM had to generate Kernel/IR
        model (str, optional): _description_. Defaults to "llama3.2:latest"
        outputIR (str, optional): _description_. Defaults to "CUDA"

    Returns:
        str: New version of the kernel 
    """

    
    sys_prompt = prompts.get_fixer_sys_prompt(outputIR)
    prompt = prompts.generate_fixer_prompt(kernel, error, msg)

    response = ol.chat(model=model, messages=[{"role": "system", "content": sys_prompt},{"role": "user", "content": prompt}])
    raw_output = response['message']['content']
    cleaned_output = clean_output(raw_output)

    return cleaned_output

def gemini_generator(msg: str, model: str = "gemini-2.5-flash", outputIR: str = "CUDA"):
    """Initial generation of kernel/IR using Gemini.
    ...
    """
    print("Generating code...")
    sys_prompt = prompts.get_generation_sys_prompt(outputIR)
    
    # Create the model WITH the system prompt
    chat = genai.GenerativeModel(
        model_name=model,
        system_instruction=sys_prompt  # <-- Pass system prompt here
    )
    
    # Send ONLY the user message
    response = chat.generate_content(
        [{"role": "user", "parts": msg}] # <-- No "system" role
    )

    raw_output = response.text
    cleaned_output = clean_output(raw_output)
    print("Code generated...")
    return cleaned_output

def gemini_fixer(kernel: str, error: str, msg: str, model: str = "gemini-2.5-flash", outputIR: str = "CUDA"):
    """Fixes the previously generated kernel using Gemini.
    ...
    """
    sys_prompt = prompts.get_fixer_sys_prompt(outputIR)
    prompt = prompts.generate_fixer_prompt(kernel, error, msg)
    
    # Create the model WITH the system prompt
    chat = genai.GenerativeModel(
        model_name=model,
        system_instruction=sys_prompt  # <-- Pass system prompt here
    )
    
    # Send ONLY the user message (which contains the 'prompt')
    response = chat.generate_content(
        [{"role": "user", "parts": prompt}] # <-- No "system" role
    )

    raw_output = response.text
    cleaned_output = clean_output(raw_output)
    return cleaned_output
