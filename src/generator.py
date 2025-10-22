import re
import ollama as ol
import google.generativeai as genai
from src.prompts import prompts

def parse_llm_output(raw_output: str) -> tuple[str, str]:
    """
    Parses the LLM's raw output to find the kernel.cu and wrapper.cpp blocks.
    
    Args:
        raw_output: The complete string response from the LLM.

    Returns:
        A tuple (kernel_code, wrapper_code)
        
    Raises:
        ValueError: If either the kernel or wrapper block is not found.
    """
    kernel_pattern = r"// \[START kernel\.cu\](.*?)// \[END kernel\.cu\]"
    wrapper_pattern = r"// \[START wrapper\.cpp\](.*?)// \[END wrapper\.cpp\]"
    
    kernel_match = re.search(kernel_pattern, raw_output, re.DOTALL)
    wrapper_match = re.search(wrapper_pattern, raw_output, re.DOTALL)
    
    if not kernel_match:
        raise ValueError("LLM output did not contain the expected '// [START kernel.cu]' block.")
        
    if not wrapper_match:
        raise ValueError("LLM output did not contain the expected '// [START wrapper.cpp]' block.")

    kernel_code = kernel_match.group(1).strip()
    wrapper_code = wrapper_match.group(1).strip()
    
    return kernel_code, wrapper_code

def ollama_generator(msg: str, model: str = "llama3.2:latest", outputIR: str = "CUDA") -> tuple[str, str]:
    """Initial generation of kernel/IR

    Args:
        msg (str): Context for LLM to generate Kernel/IR
        model (str, optional): Which Ollama model to use for LLM. Defaults to "llama3.2:latest".
        outputIR (str, optional): What is the desired output IR type. Defaults to "CUDA".

    Returns:
        tuple[str, str]: (kernel_code, wrapper_code)
    """
    print("Generating code...")
    sys_prompt = prompts.get_generation_sys_prompt(outputIR)
    response = ol.chat(model=model, messages=[{"role": "system", "content": sys_prompt},{"role": "user", "content": msg}])
    
    raw_output = response['message']['content']
    cu_code, cpp_code = parse_llm_output(raw_output)
    
    print("Code generated...")
    return cu_code, cpp_code

def ollama_fixer(cu_code: str, cpp_code: str, error: str, msg: str, model: str = "llama3.2:latest", outputIR: str = "CUDA") -> tuple[str, str]:
    """Fixes the previously generated kernel using Gemini.

    Args:
        cu_code (str): Previous version of the malformed/incorrect .cu kernel
        cpp_code (str): Previous version of the malformed/incorrect .cpp wrapper
        error (str): Custom error message to inform what the LLM did wrong 
        msg (str): Context the ORIGINAL LLM had to generate Kernel/IR

    Returns:
        tuple[str, str]: (new_kernel_code, new_wrapper_code)
    """

    sys_prompt = prompts.get_fixer_sys_prompt(outputIR)
    
    prompt = prompts.generate_fixer_prompt(cu_code, cpp_code, error, msg)

    response = ol.chat(model=model, messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ])
    
    raw_output = response['message']['content']
    new_cu_code, new_cpp_code = parse_llm_output(raw_output)

    return new_cu_code, new_cpp_code

def gemini_generator(msg: str, model: str = "gemini-2.5-flash", outputIR: str = "CUDA") -> tuple[str, str]:
    """Initial generation of kernel/IR using Gemini.

    Returns:
        tuple[str, str]: (kernel_code, wrapper_code)
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

    raw_output = response.text
    cu_code, cpp_code = parse_llm_output(raw_output)
    
    print("Code generated...")
    return cu_code, cpp_code

def gemini_fixer(cu_code: str, cpp_code: str, error: str, msg: str, model: str = "gemini-2.5-flash", outputIR: str = "CUDA") -> tuple[str, str]:
    """Fixes the previously generated kernel using Gemini.

    Args:
        cu_code (str): Previous version of the malformed/incorrect .cu kernel
        cpp_code (str): Previous version of the malformed/incorrect .cpp wrapper
        error (str): Custom error message to inform what the LLM did wrong 
        msg (str): Context the ORIGINAL LLM had to generate Kernel/IR

    Returns:
        tuple[str, str]: (new_kernel_code, new_wrapper_code)
    """
    sys_prompt = prompts.get_fixer_sys_prompt(outputIR)
    
    prompt = prompts.generate_fixer_prompt(cu_code, cpp_code, error, msg)
    
    chat = genai.GenerativeModel(
        model_name=model,
        system_instruction=sys_prompt
    )
    
    response = chat.generate_content(
        [{"role": "user", "parts": [prompt]}]
    )

    raw_output = response.text
    new_cu_code, new_cpp_code = parse_llm_output(raw_output)
    
    return new_cu_code, new_cpp_code