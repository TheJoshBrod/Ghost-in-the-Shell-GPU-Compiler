import re
import ollama as ol
import google.generativeai as genai
#from openai import OpenAI
from src.prompts import prompts

def cleanup_mkdown(input: str) -> str:
    """Extract code from markdown code blocks using regex."""

    # Try to match code blocks with language specifiers (C++, cpp, cuda, c)
    pattern = r"```(?:C\+\+|cpp|cuda|c)\s*\n(.*?)```"
    match = re.search(pattern, input, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Try generic code block without language specifier
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, input, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # No markdown found, return as-is
    return input.strip()


def ollama_generator(msg: str, model: str = "llama3.2:latest", outputIR: str = "CUDA") -> str:
    """Initial generation of kernel/IR

    Args:
        msg (str): Context for LLM to generate Kernel/IR
        model (str, optional): Which Ollama model to use for LLM. Defaults to "llama3.2:latest".
        outputIR (str, optional): What is the desired output IR type. Defaults to "CUDA".

    Returns:
        str: kernel_code
    """
    print("Generating code...")
    sys_prompt = prompts.get_generation_sys_prompt(outputIR)
    response = ol.chat(model=model, messages=[{"role": "system", "content": sys_prompt},{"role": "user", "content": msg}])
    
    cu_code = response['message']['content']
    
    
    print("Code generated...")
    return cleanup_mkdown(cu_code)

def ollama_fixer(cu_code: str, error: str, msg: str, model: str = "llama3.2:latest", outputIR: str = "CUDA") -> str:
    """Fixes the previously generated kernel using Gemini.

    Args:
        cu_code (str): Previous version of the malformed/incorrect .cu kernel
        error (str): Custom error message to inform what the LLM did wrong 
        msg (str): Context the ORIGINAL LLM had to generate Kernel/IR

    Returns:
        str: new_kernel_code
    """

    sys_prompt = prompts.get_fixer_sys_prompt(outputIR)
    
    prompt = prompts.generate_fixer_prompt(cu_code, error, msg)

    response = ol.chat(model=model, messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ])
    
    new_cu_code = response['message']['content']

    return cleanup_mkdown(new_cu_code)

def gemini_generator(msg: str, model: str = "gemini-2.5-flash", outputIR: str = "CUDA") -> str:
    """Initial generation of kernel/IR using Gemini.

    Returns:
        str: kernel_code
    """
    print("Generating code...")
    sys_prompt = prompts.get_generation_sys_prompt(outputIR)
    
    chat = genai.GenerativeModel(
        model_name=model,
        system_instruction=sys_prompt
    )
    
    response = chat.generate_content(
        [{"role": "user", "parts": [msg]}]
    )
    cu_code = cleanup_mkdown(response.text)

    print("Code generated...")
    return cu_code

def gemini_fixer(cu_code: str, error: str, msg: str, model: str = "gemini-2.5-flash", outputIR: str = "CUDA") -> str:
    """Fixes the previously generated kernel using Gemini.

    Args:
        cu_code (str): Previous version of the malformed/incorrect .cu kernel
        error (str): Custom error message to inform what the LLM did wrong 
        msg (str): Op details from initial run
        model (str): Gemini model
        outputIR (str): what output we want

    Returns:
        str: new_kernel_code
    """
    sys_prompt = prompts.get_fixer_sys_prompt(outputIR)
    
    prompt = prompts.generate_fixer_prompt(cu_code, error, msg)
    
    chat = genai.GenerativeModel(
        model_name=model,
        system_instruction=sys_prompt
    )
    
    response = chat.generate_content(
        [{"role": "user", "parts": [prompt]}]
    )

    new_cu_code = cleanup_mkdown(response.text)
    
    return new_cu_code

def chatgpt_generator(msg: str, model: str = "gpt-4o", outputIR: str = "CUDA") -> str:
    """Initial generation of kernel/IR using OpenAI.

    Returns:
        str: kernel_code
    """

    client = OpenAI()
        
    print("Generating code...")
    sys_prompt = prompts.get_generation_sys_prompt(outputIR)
    
    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": msg}
            ]
        )

    cu_code = cleanup_mkdown(response.choices[0].message.content)
    
    print("Code generated...")
    return cu_code
        

def chatgpt_fixer(cu_code: str, error: str, msg: str, model: str = "gpt-4o", outputIR: str = "CUDA") -> str:
    """Fixes the previously generated kernel using OpenAI.

    Args:
        cu_code (str): Previous version of the malformed/incorrect .cu kernel
        error (str): Custom error message to inform what the LLM did wrong 
        msg (str): Op details from initial run
        model (str): OpenAI model
        outputIR (str): what output we want

    Returns:
        str: new_kernel_code
    """
    client = OpenAI()
        
    sys_prompt = prompts.get_fixer_sys_prompt(outputIR)
    prompt = prompts.generate_fixer_prompt(cu_code, error, msg)
    
    response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
        )

    cu_code = cleanup_mkdown(response.choices[0].message.content)
    
    print("Code generated...")
    return cu_code
