import litellm
from datasets import load_dataset
import jsonlines
import dateutil

from itertools import batched
from datetime import datetime
import os
import ast
import sys
from pathlib import Path

from eval_helpers import synth_process_response, dnd_process_response
from utils import filter_by_context_length


def interpret_litellm_output(datapoint, model, output, response_processor):
    if output == litellm.BadRequestError:
        this_output = {
            "id": datapoint["id"],
            "context_window_id": datapoint["context_window_id"],
            "dataset": datapoint["dataset"],
            "model": model,
            "attempted_parse": "ERROR",
            "parse_confidence": "ERROR",
            "full_answer": "ERROR",
            "score": 0,
            "context_len": datapoint['context_len'],
            "task_group":  datapoint['task_group'],
            "task":  datapoint['task'],
            "answer_type":  datapoint['answer_type'],
            "answer": str(datapoint['answer'].strip('[').strip(']')),
        }
        print("WARNING:ERROR")
    else:
        output = output["choices"][0]["message"]["content"]
        if output is None:
            if output['choices'][0]['finish_reason'] == 'content_filter':
                output="CONTENT_FILTERED"
                print("WARNING: CONTENT FILTERED")
            else:
                raise ValueError("empty output!")
        this_output = response_processor(datapoint, output, model)
    

    return this_output


def launch(model, dataset, reasoning_level, labels, batch_by_context_window, batch_size, max_context_len, min_context_len, model_prefix, base_url):
    split_to_use="context_window_text"
    api_args = {}

    # download data
    if dataset == "synth":
        data = load_dataset(f"oolongbench/oolong-synth")["test"]
        process_response = synth_process_response
    else:
        data = load_dataset(f"oolongbench/oolong-real", 'dnd')["test"]
        process_response = dnd_process_response
        data = filter_by_context_length(data, model)

    results_dir = "results"
    do_cache = True
    safemodelname = model.split("/")[-1]# +"-labels"

    if labels:
        safemodelname += "-labels"
        split_to_use="context_window_text_with_labels"
    
    if reasoning_level != "":
        safemodelname += f"-{reasoning_level}"
        api_args['reasoning_effort'] = reasoning_level
        api_args['extra_body'] = { 
            "allowed_openai_params": ["reasoning_effort"]
        }

    Path(f"{results_dir}/{dataset}/{safemodelname}").mkdir(parents=True, exist_ok=True)

    # sort by context window ID (to enable caching)
    data = data.sort("context_window_id")


    # for each example:
    output_counter = 0
    correct = 0


    all_outputs = []
    current_outputs = []


    data = data.filter(lambda x: x['context_len'] <= max_context_len) 
    data = data.filter(lambda x: x['context_len'] > min_context_len) 
    print(f"Evaluating {len(data)} examples for model {model} with context lengths between {min_context_len} and {max_context_len}...")

    # potentially init from prior partial run
    full_results_path = f"{results_dir}/{dataset}/{safemodelname}/full_output.jsonl"
    if os.path.exists(full_results_path):
        ids_to_skip = []
        correct = 0
        output_counter = 0
        with jsonlines.open(full_results_path, 'r') as f:
            for obj in f:
                ids_to_skip.append(obj['id'])
                correct += obj['score']
                output_counter += 1
        data = data.filter(lambda x: x['id'] not in ids_to_skip)
        print(f"Caution: filtered out completed examples; {len(data)} examples left to run...")
    else:
        with jsonlines.open(full_results_path, 'w') as f:
            pass # init file

    try:
        print("starting data")
        if batch_by_context_window:
            context_windows = list(set(data['context_window_id']))
            for context_window_id in context_windows:
                this_windowset_data = data.filter(lambda x: x['context_window_id'] == context_window_id)
                this_window_data = this_windowset_data
                score = 0

                if do_cache:
                    # run one through, then run the remainder as a batch
                    seed_datapoint = this_window_data[0]

                    datapoint = seed_datapoint
                    client = litellm.completion(
                        api_key=os.environ.get("LITELLM_API_KEY"),
                        base_url=base_url,
                        tools=[],
                        model=f"{model_prefix}{model}",
                        api_version="2024-12-01",
                        extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
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
                                        "cache_control": {"type": "ephemeral"},
                                    },
                                ]
                            },
                            {
                                "role": "user",
                                "content": f"{seed_datapoint['question']}",
                            },
                        ],
                        **api_args,
                    )
                    output = client["choices"][0]["message"]["content"]

                    this_output = interpret_litellm_output(datapoint, model, client[count], process_response)

                    all_outputs.append(this_output)
                    current_outputs.append(this_output)
                
                    correct += this_output['score']
                    output_counter += 1
                
                    
                    # now batch the rest
                    # convert the dictionary of lists to a list of dictionaries 
                    # fancy pythonic way to do this: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
                    remaining_window_examples = [dict(zip(this_window_data[1:],t)) for t in zip(*this_window_data[1:].values())]
                    print(f"Cache call done. Running {len(remaining_window_examples)} more examples as a single batch...")
                
                else:
                    # all examples are remaining
                    remaining_window_examples = [dict(zip(this_window_data[0:],t)) for t in zip(*this_window_data[1:].values())]
                    print(f"Running window with {len(remaining_window_examples)} examples...")

                client = litellm.batch_completion(
                    api_key=os.environ.get("LITELLM_API_KEY"),
                    base_url=base_url,
                    model=f"{model_prefix}{model}",
                    extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
                    #max_completion_tokens=32768,
                    tools=[],
                    messages = [
                        [{
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You are a helpful assistant.",
                                },
                                {
                                    "type": "text",
                                    "text": f"{datapoint[split_to_use]}",
                                    #"cache_control": {"type": "ephemeral"},
                                },
                            ]
                        },
                        {
                            "role": "user",
                            "content": f"{datapoint['question']}",
                        },
                        ] for datapoint in remaining_window_examples
                    ],
                    **api_args,

                )
                for count, datapoint in enumerate(remaining_window_examples):
                    this_output = interpret_litellm_output(datapoint, model, client[count], process_response)
                    correct += this_output['score']
                    output_counter += 1
                    all_outputs.append(this_output)
                    current_outputs.append(this_output)

                # save partial and full output for this batch

                with jsonlines.open(
                    f"{results_dir}/{safemodelname}/full_output.jsonl", "a"
                ) as f:
                    for line in current_outputs:
                        f.write(line)
                with jsonlines.open(
                    f"{results_dir}/{safemodelname}/partial_output_{output_counter - len(this_window_data)}_{output_counter - 1}.jsonl",
                    "w",
                ) as f:
                    for line in current_outputs:
                        f.write(line)
                    print(f"score so far: {correct}/{output_counter}")
                current_outputs = []
        else:
            for batch in batched(data, batch_size):
                client = litellm.batch_completion(
                    api_key=os.environ.get("LITELLM_API_KEY"),
                    base_url=base_url,
                    extra_headers={"anthropic-beta": "context-1m-2025-08-07"},
                    model=f"{model_prefix}{model}",
                    tools=[],
                    messages = [
                        [{
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "You are a helpful assistant.",
                                },
                                {
                                    "type": "text",
                                    "text": f"{datapoint[split_to_use]}",
                                    "cache_control": {"type": "ephemeral"},
                                },
                            ]
                        },
                        {
                            "role": "user",
                            "content": f"{datapoint['question']}",
                        },
                        ] for datapoint in batch
                    ],
                    **api_args,

                )
                for count, datapoint in enumerate(batch):
                    this_output = interpret_litellm_output(datapoint, model, client[count], process_response)
                    correct += this_output['score']
                    output_counter += 1
                    all_outputs.append(this_output)
                    current_outputs.append(this_output)
                # save partial and full output for this batch

                with jsonlines.open(
                    f"{results_dir}/{safemodelname}/full_output.jsonl", "a"
                ) as f:
                    for line in current_outputs:
                        f.write(line)
                with jsonlines.open(
                    f"{results_dir}/{safemodelname}/partial_output_{output_counter - len(batch)}_{output_counter - 1}.jsonl",
                    "w",
                ) as f:
                    for line in current_outputs:
                        f.write(line)
                    print(f"score so far: {correct}/{output_counter}")
                current_outputs = []


    except Exception as e:
        error_file_loc = f"{results_dir}/{safemodelname}/error_partial_results.jsonl"
        with jsonlines.open(error_file_loc, "w") as f:
            for line in all_outputs:
                f.write(line)

        print(client)
        raise ValueError(
            f"Error on datapoint {datapoint['id']}, which is item {output_counter}. Saved partial results to {error_file_loc}. Error: {e}"
        )


    with jsonlines.open(f"{results_dir}/{safemodelname}/full_output.jsonl", "a") as f:
        for line in current_outputs:
            f.write(line)

    with open(f"{results_dir}/{safemodelname}/overall.txt", "w") as f:
        summary = f"Overall score for {model} on {len(data)} examples: {correct}/{len(data)} = {correct / len(data)}"
        print(summary)
        f.write(summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    

    parser.add_argument(
        '--dataset', 
        default='synth', 
        choices=['synth', 'real'],
        help='Dataset to use (default: synth)'
    )

    parser.add_argument(
        '--reasoning_level', 
        default='', 
        choices=['', 'low', 'medium', 'high', 'minimal'],
        help='Reasoning level (default: empty string)'
    )
    
    parser.add_argument(
        '--labels', 
        action='store_true',
        default=False,
        help='Enable labels (default: False)'
    )

    parser.add_argument(
        '--batch_by_context_window', 
        action='store_true',
        default=False,
        help='Enable batching by context window (default: False)'
    )

    parser.add_argument(
        '--batch_size', 
        default=1,
        type=int,
        help='number of examples to run at once'
    )
    
    parser.add_argument(
        '--max_context_len', 
        default=131072,
        type=int,
        help='max context length to include'
    )


    parser.add_argument(
        '--min_context_len', 
        default=1024,
        type=int,
        help='min context length to include'
    )


    parser.add_argument(
        '--model_prefix', 
        default="",
        type=str,
        help="a prefix to append to all models (e.g. 'litellm_proxy/')"
    )
    parser.add_argument(
        '--base_url', 
        default=None,
        type=str,
        help="a base URL for a hosted litellm instance, if necessary"
    )

    parser.add_argument(
        '--model', 
        required=True,
        help='Model name (required)'
    )
    
    args = parser.parse_args()

    launch(args.model, args.dataset, args.reasoning_level, args.labels, args.batch_by_context_window, args.batch_size, args.max_context_len, args.min_context_len, args.model_prefix, args.base_url)