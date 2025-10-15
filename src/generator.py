import regex as re
import ollama as ol
from fastapi import FastAPI

from src.prompts import prompts

app = FastAPI()

def clean_output(raw: str) -> str:
    """Removes ```CUDA, ```, and other markdown fluff."""

    # Remove the suffix first
    if raw.endswith("```"):
        raw = raw[:-3]  # remove last 3 chars
    
    # Remove the prefix like ```cuda
    raw = re.sub(r"^```(\w+)?\n", "", raw)

    return raw

@app.get("/ollama")
def ollama_endpoint(msg: str, inputs: dict, model: str = "llama3.2:latest", outputIR: str = "CUDA"):

    # Generate initial kernel
    sys_prompt = prompts.get_sys_prompt(outputIR)
    response = ol.chat(model=model, messages=[{"role": "system", "content": sys_prompt},{"role": "user", "content": msg}])
    raw_output = response['message']['content']
    cleaned_output = clean_output(raw_output)

    # Handle Validation
    # TODO

    # Handle feedback loop
    # TODO

    final_output = cleaned_output

    return {"kernel": final_output}