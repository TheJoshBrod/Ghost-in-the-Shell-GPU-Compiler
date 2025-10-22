# GinS: Ghost in the Shell

**GinS** is an automated system for generating and validating CUDA kernels from PyTorch operations. It uses AI-powered iterative refinement to convert high-level PyTorch operations into optimized CUDA kernels with comprehensive validation.

## Features

- **Automated CUDA Kernel Generation**: Converts PyTorch operations to equivalent CUDA kernels
- **Iterative Validation & Refinement**: AI-powered error fixing with up to 5 attempts per operation
- **Multi-LLM Support**: Works with Google Gemini and Ollama models
- **Complete PyTorch Integration**: Generates kernels as PyTorch C++ extensions
- **Robust Error Handling**: Comprehensive validation covering compilation and correctness
- **Benchmark Processing**: Handles sequences of operations with shared execution context
- **Detailed Logging**: Saves all attempts, feedback, and results for analysis

## Architecture

The system follows a 3-stage pipeline for each operation:

1. **Monitor** (`src/monitor.py`): Profiles PyTorch operations to capture ATen calls and CUDA kernel information
2. **Generate** (`src/generator.py`): Uses LLMs to generate CUDA kernel code based on profiling data
3. **Verify** (`src/verifier.py`): Compiles and validates generated kernels against ground truth
## Requirements

### Core Software
- **Python** ≥ 3.12  
- **PyTorch** (CUDA-enabled build)  
- **NVIDIA GPU** with CUDA support  
- **LLM Provider of your choice**:
    - **Google Gemini API key** - required for Gemini-based models  
    - **Ollama** - required for running local models  
    - **OpenAI API key** - required for OpenAI-based models

### Minimum Versions

| **Component** | **Minimum Version** | **Reason / Notes** |
|----------------|--------------------|--------------------|
| **NVCC** | ≥ 12.1 | Required for full C++17 support with PyTorch 2.x |
| **GCC** | ≥ 11.x | Compatible with NVCC 12.1 toolchain |
| **PyTorch** | ≥ 2.0 | Required for modern extension APIs and C++17 |
| **CUDA Driver** | ≥ 12.0 | Must support CUDA 12.x toolkit |

** If issue with compiling generated code, likely compatibility issue with one of these (update them). 

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd GinS
   ```

2. **Install dependencies:**
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up API keys (for Gemini):**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

4. **Install Ollama (optional, for local models):**
   ```bash
   # Follow Ollama installation instructions for your platform
   ollama pull llama3.2:latest
   ```

## Usage

### Basic Usage

Run the system on a benchmark file:

```bash
python -m src.main benchmarks/initial_testing.json
```

### Benchmark Format

Create a JSON file with the following structure:

```json
[
    {
        "name": "program1",
        "operations": [
            {
                "assignment": "c",
                "operation": "torch.matmul",
                "inputs": ["a", "b"]
            },
            {
                "assignment": "d",
                "operation": "torch.sin",
                "inputs": ["c"]
            }
        ],
        "definitions": [
            {"variable": "a", "value": "torch.randn(2048, 2048, device=\"cuda\")"},
            {"variable": "b", "value": "torch.randn(2048, 2048, device=\"cuda\")"}
        ]
    }
]
```

### Configuration

Key configuration options in `src/main.py`:

- `MAX_ATTEMPTS = 5`: Maximum retry attempts per operation
- `OUTPUT_DIR = "generated_kernels"`: Directory for output files

## Project Structure

```
GinS/
├── src/
│   ├── main.py              # Main pipeline orchestrator
│   ├── monitor.py           # PyTorch operation profiling
│   ├── generator.py         # LLM-based code generation
│   ├── verifier.py          # Kernel validation and testing
│   └── prompts/
│       └── prompts.py       # LLM system prompts
├── benchmarks/
│   └── initial_testing.json # Example benchmark file
├── generated_kernels/       # Output directory for results
├── requirements.txt         # Python dependencies
└── run.sh                   # Example run script
```

## Workflow

1. **Load Benchmark**: Parse JSON file with operations and variable definitions
2. **Initialize Context**: Set up execution environment with defined variables
3. **Process Operations**: For each operation:
   - **Profile**: Monitor PyTorch execution to capture ATen/kernel information
   - **Generate**: Use LLM to create CUDA kernel code
   - **Validate**: Compile and test against ground truth
   - **Refine**: If validation fails, use AI to fix errors and retry (default: up to 5 attempts)
4. **Save Results**: Store final kernels and logs

## Output Files

The system generates several types of output files:

- `*_inputs.pt`: Input tensors for validation
- `*_gold.pt`: Ground truth output tensors
- `*_iter{N}.log`: Logs for each validation attempt
- `*_kernel_final.cu`: Final generated kernel (successful)
- `*_kernel_final_FAILED.cu`: Final kernel (if all attempts failed)

## Example

The included `benchmarks/initial_testing.json` demonstrates:

- Matrix multiplication (`torch.matmul`)
- Element-wise operations (`torch.sin`)
- Sequential operations with shared context

## Troubleshooting

### Common Issues

1. **CUDA not available**: Ensure PyTorch is installed with CUDA support
2. **Compilation errors**: Check that generated kernels follow PyTorch C++ extension requirements (NVCC, GCC, etc.)
3. **API key issues**: Verify API key is set correctly for models OR ollama running in separate terminal for local models
4. **Memory issues**: Reduce tensor sizes in benchmark definitions

### Debug Mode

Enable detailed logging by modifying the logging level in `src/verifier.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```
