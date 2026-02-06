#!/usr/bin/env python3
"""Launch a vLLM server for a given model.

This script starts a vLLM OpenAI-compatible server that can be used with
the eval script via the litellm backend with a custom base_url.

Example usage:
    python launch_vllm_server.py --model meta-llama/Llama-3-8B-Instruct
    python launch_vllm_server.py --model meta-llama/Llama-3-8B-Instruct --port 8080 --tensor-parallel-size 2
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch a vLLM OpenAI-compatible server"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length (default: model's max)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float32"],
        help="Data type for model weights (default: auto)",
    )

    args = parser.parse_args()

    # Build the vllm serve command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
    ]

    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    # Print connection info
    print("=" * 60)
    print("Starting vLLM OpenAI-compatible server")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Server URL: http://{args.host}:{args.port}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print()
    print("To use with eval_script_batched.py:")
    print("-" * 60)
    print(f"  python eval_script_batched.py \\")
    print(f"      --model {args.model} \\")
    print(f"      --backend litellm \\")
    print(f"      --base_url http://localhost:{args.port}/v1 \\")
    print(f"      --model_prefix 'openai/' \\")
    print(f"      --dataset synth")
    print()
    print("Or use direct vLLM inference (no server needed):")
    print("-" * 60)
    print(f"  python eval_script_batched.py \\")
    print(f"      --model {args.model} \\")
    print(f"      --backend vllm_local \\")
    print(f"      --dataset synth")
    print()
    print("=" * 60)
    print("Server starting... (Ctrl+C to stop)")
    print("=" * 60)
    print()

    # Launch the server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nServer exited with error code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\nError: vLLM is not installed. Install it with: pip install vllm")
        sys.exit(1)


if __name__ == "__main__":
    main()
