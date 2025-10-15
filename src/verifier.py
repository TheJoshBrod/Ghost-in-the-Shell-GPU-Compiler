"""Iterative toolchain to take different IRs with input and outputs and """

import tempfile
import subprocess
import os
import shutil
import uuid
import textwrap

def _normalize_output(s: str) -> str:
    # Normalize whitespace/newlines for comparison. Adjust as needed.
    return "\n".join(line.rstrip() for line in s.strip().splitlines())

def _static_cuda_checks(code: str) -> str:
    """Return non-empty message explaining what's missing or suspicious, or empty if looks okay."""
    messages = []
    lowered = code  # keep case for identifiers; CUDA keywords are lowercase anyway
    if "__global__" not in lowered and "__device__" not in lowered and "__host__" not in lowered:
        messages.append("No CUDA function qualifiers found (e.g. '__global__', '__device__', '__host__').")
    if "<<<" not in lowered or ">>>" not in lowered:
        messages.append("No kernel-launch syntax '<<<...>>>' detected.")
    if "#include <cuda.h>" not in lowered and "#include <cuda_runtime.h>" not in lowered:
        # not strictly necessary but helpful to warn
        messages.append("No obvious CUDA runtime include (e.g. <cuda_runtime.h>).")
    if "cudaMemcpy" not in lowered and "cudaMalloc" not in lowered and "cudaFree" not in lowered:
        messages.append("No calls to cudaMalloc/cudaMemcpy/cudaFree detected. Maybe code is missing host-device data movement.")
    # Provide very lightweight check for main
    if "int main(" not in lowered and "void main(" not in lowered:
        messages.append("No 'main' function detected; compilation may fail or the binary may do nothing.")
    return "\n".join(messages)

def verify_cuda(code: str, correct_output: str, timeout_seconds: int = 10) -> list[bool, str]:
    """
    Try to compile and run provided CUDA C/C++ source using nvcc (if available),
    compare stdout to correct_output, and return (matches: bool, message: str).

    - If nvcc is present: returns (True, "") when output matches exactly (after normalization),
      otherwise (False, "<compiler/runtime/diff message>").
    - If nvcc is NOT present: performs static checks and returns (False, "<advice>") unless
      static checks are all OK, in which case returns (False, "<nvcc missing but code looks plausible>").

    Note: exact string match is used after normalization. Adjust _normalize_output if you need fuzzy matching.
    """

    # TODO: Go through all of this.

    # Quick sanity
    if not isinstance(correct_output, str):
        return False, "Provided 'correct_output' is not a string."

    # NVCC not installed
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        # Do static checks and return helpful info
        static_msg = _static_cuda_checks(code)
        if static_msg:
            return False, "nvcc not found on PATH. Static analysis found potential issues:\n" + static_msg
        else:
            return False, "nvcc not found on PATH. Source looks plausible (no obvious issues), but I can't compile/run it here."

    # NVCC is installed: compile & run
    tmpdir = tempfile.mkdtemp(prefix="verify_cuda_")
    try:
        src_path = os.path.join(tmpdir, f"prog_{uuid.uuid4().hex}.cu")
        bin_path = os.path.join(tmpdir, "prog_exec")

        # write source
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)

        # compile source code
        compile_cmd = [nvcc_path, src_path, "-o", bin_path]
        try:
            compile_proc = subprocess.run(
                compile_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=60
            )
        except subprocess.TimeoutExpired:
            return False, "nvcc compilation timed out."
        
        # compile failed
        if compile_proc.returncode != 0:
            # return compiler stderr for debugging
            err = compile_proc.stderr.strip()
            out = compile_proc.stdout.strip()
            msg = "Compilation failed.\n"
            if err:
                msg += f"nvcc stderr:\n{err}\n"
            if out:
                msg += f"nvcc stdout:\n{out}\n"
            return False, msg

        
        # run the binary, capture stdout
        exec_path = bin_path
        try:
            run_proc = subprocess.run(
                [exec_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired:
            return False, f"Program execution timed out after {timeout_seconds} seconds."

        # If program returned non-zero, still capture stderr
        stdout = run_proc.stdout or ""
        stderr = run_proc.stderr or ""
        if run_proc.returncode != 0:
            msg = f"Program exited with return code {run_proc.returncode}.\n"
            if stderr:
                msg += f"stderr:\n{stderr}\n"
            msg += f"stdout:\n{stdout}\n"
            return False, msg

        # Compare outputs after normalization
        got = _normalize_output(stdout)
        want = _normalize_output(correct_output)
        if got == want:
            return True, ""
        else:
            # Provide helpful diff-like message (simple)
            msg = "Output mismatch.\n--- got ---\n"
            msg += got + "\n--- expected ---\n" + want + "\n"
            if stderr:
                msg += "\nProgram stderr:\n" + stderr
            return False, msg

    finally:
        # clean up temporary files; swallow exceptions
        try:
            for fname in os.listdir(tmpdir):
                path = os.path.join(tmpdir, fname)
                try:
                    os.remove(path)
                except Exception:
                    pass
            os.rmdir(tmpdir)
        except Exception:
            pass
