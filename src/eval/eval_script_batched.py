import os
import warnings
from itertools import batched
from pathlib import Path

import jsonlines
import litellm
from datasets import load_dataset
from eval_helpers import dnd_process_response, synth_process_response
from utils import compute_context_lengths

# Optional dependencies for local backends
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class ModelError:
    """Unified error sentinel for all backends."""

    def __init__(self, message="Unknown error"):
        self.message = message

    def __repr__(self):
        return f"ModelError({self.message})"


def init_backend(backend, model):
    """Initialize the backend and return model/tokenizer objects.

    For litellm, returns None (no initialization needed).
    For vllm_local, returns the LLM object.
    For hf, returns a tuple of (model, tokenizer).
    """
    if backend == "litellm":
        return None

    elif backend == "vllm_local":
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vllm is not installed. Install it with: pip install vllm"
            )
        print(f"Initializing vLLM with model {model}...")
        llm = LLM(model=model)
        return llm

    elif backend == "hf":
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers is not installed. Install it with: pip install transformers torch"
            )
        print(f"Initializing HuggingFace model {model}...")
        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return (hf_model, tokenizer)

    else:
        raise ValueError(f"Unknown backend: {backend}")


def format_messages_to_prompt(messages, tokenizer=None):
    """Convert OpenAI-style messages to a single prompt string.

    If tokenizer has a chat template, use it. Otherwise, format manually.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback: manual formatting
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            # Handle structured content (list of dicts with "text" keys)
            text_parts = [
                part["text"] for part in content if isinstance(part, dict) and "text" in part
            ]
            content = "\n".join(text_parts)
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


def make_openai_response(content, finish_reason="stop"):
    """Create an OpenAI-compatible response dict."""
    return {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish_reason,
            }
        ]
    }


def call_model_litellm(
    messages, model, model_prefix, base_url, api_args, use_cache=False
):
    """Call model using litellm."""
    if use_cache and messages:
        # Add cache_control to system message content
        system_msg = messages[0]
        if system_msg["role"] == "system" and isinstance(system_msg["content"], list):
            # Already has structured content, add cache_control to last text block
            for part in reversed(system_msg["content"]):
                if isinstance(part, dict) and "text" in part:
                    part["cache_control"] = {"type": "ephemeral"}
                    break

    try:
        response = litellm.completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url=base_url,
            tools=[],
            model=f"{model_prefix}{model}",
            api_version="2024-12-01",
            extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
            messages=messages,
            **api_args,
        )
        return response
    except litellm.BadRequestError as e:
        return ModelError(f"LiteLLM BadRequestError: {e}")


def call_model_batch_litellm(
    messages_list, model, model_prefix, base_url, api_args, use_cache=False
):
    """Call model using litellm batch completion."""
    if use_cache:
        # Add cache_control to each message's system content
        for messages in messages_list:
            if messages and messages[0]["role"] == "system":
                content = messages[0]["content"]
                if isinstance(content, list):
                    for part in reversed(content):
                        if isinstance(part, dict) and "text" in part:
                            part["cache_control"] = {"type": "ephemeral"}
                            break

    try:
        responses = litellm.batch_completion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url=base_url,
            model=f"{model_prefix}{model}",
            extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
            tools=[],
            messages=messages_list,
            **api_args,
        )
        return responses
    except litellm.BadRequestError as e:
        return [ModelError(f"LiteLLM BadRequestError: {e}")] * len(messages_list)


def call_model_vllm(messages, client, tokenizer=None):
    """Call model using vLLM direct inference."""
    try:
        # vLLM's LLM object has a tokenizer we can use
        if tokenizer is None:
            tokenizer = client.get_tokenizer()

        prompt = format_messages_to_prompt(messages, tokenizer)

        # Calculate max tokens based on model context length minus input
        input_ids = tokenizer.encode(prompt)
        model_max_len = client.llm_engine.model_config.max_model_len
        max_tokens = model_max_len - len(input_ids)

        sampling_params = SamplingParams(max_tokens=max_tokens)

        outputs = client.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text
        finish_reason = outputs[0].outputs[0].finish_reason

        return make_openai_response(output_text, finish_reason)
    except Exception as e:
        return ModelError(f"vLLM error: {e}")


def call_model_batch_vllm(messages_list, client, tokenizer=None):
    """Call model using vLLM batch inference."""
    try:
        if tokenizer is None:
            tokenizer = client.get_tokenizer()

        prompts = [format_messages_to_prompt(msgs, tokenizer) for msgs in messages_list]

        # Calculate max tokens for each prompt based on model context length
        model_max_len = client.llm_engine.model_config.max_model_len
        sampling_params_list = []
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt)
            max_tokens = model_max_len - len(input_ids)
            sampling_params_list.append(SamplingParams(max_tokens=max_tokens))

        outputs = client.generate(prompts, sampling_params_list)

        responses = []
        for output in outputs:
            output_text = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            responses.append(make_openai_response(output_text, finish_reason))

        return responses
    except Exception as e:
        return [ModelError(f"vLLM error: {e}")] * len(messages_list)


def call_model_hf(messages, client):
    """Call model using HuggingFace transformers."""
    try:
        model, tokenizer = client

        prompt = format_messages_to_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Calculate max new tokens based on model's context length minus input
        input_length = inputs["input_ids"].shape[1]
        model_max_len = getattr(model.config, "max_position_embeddings", None)
        if model_max_len is None:
            raise ValueError(
                "Could not determine model's max context length. "
                "Model config does not have 'max_position_embeddings'."
            )
        max_new_tokens = model_max_len - input_length

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens (exclude input)
        output_text = tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )

        finish_reason = "stop"
        if outputs.shape[1] - input_length >= max_new_tokens:
            finish_reason = "length"

        return make_openai_response(output_text, finish_reason)
    except Exception as e:
        return ModelError(f"HuggingFace error: {e}")


def call_model_batch_hf(messages_list, client):
    """Call model using HuggingFace transformers with batching."""
    try:
        model, tokenizer = client

        prompts = [format_messages_to_prompt(msgs, tokenizer) for msgs in messages_list]

        # Tokenize with padding for batching
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        # Calculate max new tokens based on longest input in batch
        model_max_len = getattr(model.config, "max_position_embeddings", None)
        if model_max_len is None:
            raise ValueError(
                "Could not determine model's max context length. "
                "Model config does not have 'max_position_embeddings'."
            )
        max_input_length = inputs["attention_mask"].sum(dim=1).max().item()
        max_new_tokens = model_max_len - max_input_length

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = []
        for i, output in enumerate(outputs):
            # Find where the input ends for this example
            input_length = (inputs["attention_mask"][i] == 1).sum().item()
            output_text = tokenizer.decode(
                output[input_length:], skip_special_tokens=True
            )

            finish_reason = "stop"
            if len(output) - input_length >= max_new_tokens:
                finish_reason = "length"

            responses.append(make_openai_response(output_text, finish_reason))

        return responses
    except Exception as e:
        return [ModelError(f"HuggingFace error: {e}")] * len(messages_list)


def interpret_output(datapoint, model, output, response_processor):
    if isinstance(output, ModelError):
        this_output = {
            "id": datapoint["id"],
            "context_window_id": datapoint["context_window_id"],
            "dataset": datapoint["dataset"],
            "model": model,
            "attempted_parse": "ERROR",
            "parse_confidence": "ERROR",
            "full_answer": "ERROR",
            "score": 0,
            "context_len": datapoint["context_len"],
            "task_group": datapoint["task_group"],
            "task": datapoint["task"],
            "answer_type": datapoint["answer_type"],
            "answer": str(datapoint["answer"].strip("[").strip("]")),
        }
        print(f"WARNING: {output}")
    else:
        content = output["choices"][0]["message"]["content"]
        if content is None:
            if output["choices"][0]["finish_reason"] == "content_filter":
                content = "CONTENT_FILTERED"
                print("WARNING: CONTENT FILTERED")
            else:
                raise ValueError("empty output!")
        this_output = response_processor(datapoint, content, model)

    return this_output


def launch(
    model,
    dataset,
    reasoning_level,
    labels,
    batch_by_context_window,
    batch_size,
    max_context_len,
    min_context_len,
    model_prefix,
    base_url,
    backend,
):
    split_to_use = "context_window_text"
    api_args = {}

    # Initialize backend and set up function pointers
    client = init_backend(backend, model)

    if backend == "litellm":
        def call_model(messages, use_cache=False):
            return call_model_litellm(
                messages, model, model_prefix, base_url, api_args, use_cache=use_cache
            )

        def call_model_batch(messages_list, use_cache=False):
            return call_model_batch_litellm(
                messages_list, model, model_prefix, base_url, api_args, use_cache=use_cache
            )
    elif backend == "vllm_local":
        def call_model(messages, use_cache=False):
            return call_model_vllm(messages, client)

        def call_model_batch(messages_list, use_cache=False):
            return call_model_batch_vllm(messages_list, client)
    elif backend == "hf":
        def call_model(messages, use_cache=False):
            return call_model_hf(messages, client)

        def call_model_batch(messages_list, use_cache=False):
            return call_model_batch_hf(messages_list, client)

    # Warn about unsupported features for local backends
    if backend in ("vllm_local", "hf") and batch_by_context_window:
        warnings.warn(
            "batch_by_context_window is not supported for local backends. "
            "Falling back to regular batching."
        )
        batch_by_context_window = False

    # download data
    if dataset == "synth":
        data = load_dataset("oolongbench/oolong-synth")["test"]
        process_response = synth_process_response
    else:
        # use 'toy_dnd' config to try out the DnD dataset
        data = load_dataset("oolongbench/oolong-real", "dnd")["test"]
        process_response = dnd_process_response
        # we compute token counts based on the model's tokenizer
        data = compute_context_lengths(data, model)

    results_dir = "results"
    do_cache = True
    safemodelname = model.split("/")[-1]  # +"-labels"

    if labels:
        safemodelname += "-labels"
        split_to_use = "context_window_text_with_labels"

    if reasoning_level != "":
        safemodelname += f"-{reasoning_level}"
        api_args["reasoning_effort"] = reasoning_level
        api_args["extra_body"] = {"allowed_openai_params": ["reasoning_effort"]}

    Path(f"{results_dir}/{dataset}/{safemodelname}").mkdir(parents=True, exist_ok=True)

    # sort by context window ID (to enable caching)
    data = data.sort("context_window_id")

    # for each example:
    output_counter = 0
    correct = 0
    total_count = 0

    all_outputs = []
    current_outputs = []

    data = data.filter(lambda x: x["context_len"] <= max_context_len)
    data = data.filter(lambda x: x["context_len"] > min_context_len)
    print(
        f"Evaluating {len(data)} examples for model {model} with context lengths between {min_context_len} and {max_context_len}..."
    )

    # potentially init from prior partial run
    full_results_path = f"{results_dir}/{dataset}/{safemodelname}/full_output.jsonl"
    if os.path.exists(full_results_path):
        ids_to_skip = []
        correct = 0
        output_counter = 0
        with jsonlines.open(full_results_path, "r") as f:
            for obj in f:
                ids_to_skip.append(obj["id"])
                correct += obj["score"]
                output_counter += 1
                total_count += 1
        data = data.filter(lambda x: x["id"] not in ids_to_skip)
        print(
            f"Caution: filtered out completed examples; {len(data)} examples left to run..."
        )
    else:
        with jsonlines.open(full_results_path, "w") as f:
            pass  # init file

    try:
        print("starting data")
        if batch_by_context_window:
            context_windows = list(set(data["context_window_id"]))
            for context_window_id in context_windows:
                this_window_data = data.filter(
                    lambda x: x["context_window_id"] == context_window_id
                )

                if do_cache:
                    # run one through, then run the remainder as a batch
                    seed_datapoint = this_window_data[0]

                    datapoint = seed_datapoint
                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You are a helpful assistant.",
                                },
                                {
                                    "type": "text",
                                    "text": f"{seed_datapoint[split_to_use]}",
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": f"{seed_datapoint['question']}",
                        },
                    ]
                    response = call_model(messages, use_cache=True)

                    this_output = interpret_output(
                        datapoint, model, response, process_response
                    )

                    all_outputs.append(this_output)
                    current_outputs.append(this_output)

                    correct += this_output["score"]
                    output_counter += 1

                    # now batch the rest
                    # convert the dictionary of lists to a list of dictionaries
                    # fancy pythonic way to do this: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
                    remaining_window_examples = [
                        dict(zip(this_window_data[1:], t))
                        for t in zip(*this_window_data[1:].values())
                    ]
                    print(
                        f"Cache call done. Running {len(remaining_window_examples)} more examples as a single batch..."
                    )

                else:
                    # all examples are remaining
                    remaining_window_examples = [
                        dict(zip(this_window_data[0:], t))
                        for t in zip(*this_window_data[1:].values())
                    ]
                    print(
                        f"Running window with {len(remaining_window_examples)} examples..."
                    )

                messages_list = [
                    [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You are a helpful assistant.",
                                },
                                {
                                    "type": "text",
                                    "text": f"{datapoint[split_to_use]}",
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": f"{datapoint['question']}",
                        },
                    ]
                    for datapoint in remaining_window_examples
                ]
                responses = call_model_batch(messages_list)

                for count, datapoint in enumerate(remaining_window_examples):
                    this_output = interpret_output(
                        datapoint, model, responses[count], process_response
                    )
                    correct += this_output["score"]
                    output_counter += 1
                    all_outputs.append(this_output)
                    current_outputs.append(this_output)

                # save partial and full output for this batch

                with jsonlines.open(
                    f"{results_dir}/{dataset}/{safemodelname}/full_output.jsonl", "a"
                ) as f:
                    for line in current_outputs:
                        f.write(line)
                with jsonlines.open(
                    f"{results_dir}/{dataset}/{safemodelname}/partial_output_{output_counter - len(this_window_data)}_{output_counter - 1}.jsonl",
                    "w",
                ) as f:
                    for line in current_outputs:
                        f.write(line)
                    print(
                        f"score so far: {correct / output_counter:.4f} ({output_counter} examples)"
                    )
                current_outputs = []
        else:
            for batch in batched(data, batch_size):
                messages_list = [
                    [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You are a helpful assistant.",
                                },
                                {
                                    "type": "text",
                                    "text": f"{datapoint[split_to_use]}",
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": f"{datapoint['question']}",
                        },
                    ]
                    for datapoint in batch
                ]
                responses = call_model_batch(messages_list, use_cache=True)

                for count, datapoint in enumerate(batch):
                    this_output = interpret_output(
                        datapoint, model, responses[count], process_response
                    )
                    correct += this_output["score"]
                    output_counter += 1
                    all_outputs.append(this_output)
                    current_outputs.append(this_output)
                # save partial and full output for this batch

                with jsonlines.open(
                    f"{results_dir}/{dataset}/{safemodelname}/full_output.jsonl", "a"
                ) as f:
                    for line in current_outputs:
                        f.write(line)
                with jsonlines.open(
                    f"{results_dir}/{dataset}/{safemodelname}/partial_output_{output_counter - len(batch)}_{output_counter - 1}.jsonl",
                    "w",
                ) as f:
                    for line in current_outputs:
                        f.write(line)
                    print(
                        f"score so far: {correct / output_counter:.4f} ({output_counter} examples)"
                    )
                current_outputs = []

    except Exception as e:
        error_file_loc = (
            f"{results_dir}/{dataset}/{safemodelname}/error_partial_results.jsonl"
        )
        with jsonlines.open(error_file_loc, "w") as f:
            for line in all_outputs:
                f.write(line)

        raise ValueError(
            f"Error on datapoint {datapoint['id']}, which is item {output_counter}. Saved partial results to {error_file_loc}. Error: {e}"
        )

    with jsonlines.open(
        f"{results_dir}/{dataset}/{safemodelname}/full_output.jsonl", "a"
    ) as f:
        for line in current_outputs:
            f.write(line)

    total_count += len(data)
    with open(f"{results_dir}/{dataset}/{safemodelname}/overall.txt", "w") as f:
        summary = f"Overall score for {model} on {total_count} examples: {correct}/{total_count} = {correct / total_count}"
        f.write(summary)
        print(summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="synth",
        choices=["synth", "real"],
        help="Dataset to use (default: synth)",
    )

    parser.add_argument(    
        "--reasoning_level",
        default="",
        choices=["", "low", "medium", "high", "minimal"],
        help="Reasoning level (default: empty string)",
    )

    parser.add_argument(
        "--labels",
        action="store_true",
        default=False,
        help="Enable labels (default: False)",
    )

    parser.add_argument(
        "--batch_by_context_window",
        action="store_true",
        default=False,
        help="Enable batching by context window (default: False)",
    )

    parser.add_argument(
        "--batch_size", default=1, type=int, help="number of examples to run at once"
    )

    parser.add_argument(
        "--max_context_len",
        default=131072,
        type=int,
        help="max context length to include",
    )

    parser.add_argument(
        "--min_context_len",
        default=1024,
        type=int,
        help="min context length to include",
    )

    parser.add_argument(
        "--model_prefix",
        default="",
        type=str,
        help="a prefix to append to all models (e.g. 'litellm_proxy/')",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        type=str,
        help="a base URL for a hosted litellm instance, if necessary",
    )

    parser.add_argument("--model", required=True, help="Model name (required)")

    parser.add_argument(
        "--backend",
        default="litellm",
        choices=["litellm", "vllm_local", "hf"],
        help="Backend to use for inference (default: litellm)",
    )

    args = parser.parse_args()

    launch(
        args.model,
        args.dataset,
        args.reasoning_level,
        args.labels,
        args.batch_by_context_window,
        args.batch_size,
        args.max_context_len,
        args.min_context_len,
        args.model_prefix,
        args.base_url,
        args.backend,
    )
