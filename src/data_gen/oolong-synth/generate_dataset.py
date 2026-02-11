# for each of the datasets: generate three hundred questions?


# iterate through all subsets of labels, selecting each subset of at least two -- maybe? or generate with a subset at each length?
# then generate questions for a context window with that subset
# then save the context window, question, gold answer for each

from datasets_loader import SUPPORTED_DATASETS
from EvalContextWindow import EvalContextWindow

import argparse
import random
import pickle
import os
import warnings

supported = [
    "trec_coarse",
    "spam",
    "imdb",
    "agnews",
    "negation",
    "yahoo",
    "formality",
    "multinli",
    "metaphors",
    "app_reviews",
]


def generate_dataset(datasetname, repeats, context_lens):
    os.makedirs('oolong-synth-datagen', exist_ok=True)

    all_data = {
        "id": [],
        "context_len": [],
        "dataset": [],
        "context_window_id": [],
        "num_labels": [],
        "context_window_text": [],
        "context_window_text_with_labels": [],
        "question": [],
        "answer": [],
        "task_group": [],
        "task": [],
        "answer_type": [],
        "input_subset": [],
    }

    id_counter = 0
    dataset_id = 100_000_000 * supported.index(datasetname)
    context_window_id = 10000 * supported.index(datasetname)

    print(f"NOW RUNNING {datasetname}")
    data = SUPPORTED_DATASETS[datasetname]()
    for context_window_size_power in context_lens:
        context_window_size = 2 ** context_window_size_power
        context_len_id = 1000000*context_window_size_power + 200
        print(f"NOW RUNNING {datasetname} AT {context_window_size}")
        for r in range(repeats):
            num_labels = len(data.label_list)
            temperature = random.randint(2, 6)
            seed = random.randint(0, 1000)

            try:
                window = EvalContextWindow(
                    data,
                    num_label_classes=num_labels,
                    context_len=context_window_size,
                    temperature=temperature,
                    seed=seed,
                )
            except Exception as e:
                print(datasetname)
                print(num_labels)
                print(context_window_size)
                print(temperature)
                print(seed)
                raise e

            def infill_example(task, current_task_group):
                nonlocal id_counter

                all_data["id"].append(dataset_id + context_len_id + context_window_id + id_counter)
                all_data["context_len"].append(context_window_size)
                all_data["dataset"].append(datasetname)
                all_data["context_window_id"].append(context_window_id)
                all_data["num_labels"].append(num_labels)
                window_text, window_text_with_labels = window.get_prompt(seed=context_window_id+context_len_id)
                all_data["context_window_text"].append(window_text)
                all_data["context_window_text_with_labels"].append(window_text_with_labels)
                all_data["question"].append(task.question)
                all_data["answer"].append(task.answer)
                all_data["task"].append(task.task_type)
                all_data["task_group"].append(current_task_group)
                all_data["answer_type"].append(task.answer_type)
                all_data["input_subset"].append(task.input_subset)
                id_counter += 1

            for task in window.counting_tasks.tasks:
                infill_example(task, "counting")
            for task in window.user_tasks.tasks:
                infill_example(task, "user")
            for task in window.temporal_tasks.tasks:
                infill_example(task, "timeline")

            context_window_id += 1
            id_counter = 0

        # reformat to fit nicely into the frame we expect

    print("Now saving")

    with open(f"oolong-synth-datagen/{datasetname}.pkl", "wb") as f:
        pickle.dump(all_data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=supported)
    parser.add_argument("--context_windows_at_each_len", type=int, default=3, help="number of context windows to create questions over")
    parser.add_argument("--context_lens", type=int, nargs="+", default=list(range(13, 17)), help="powers of 2 to use")
    args = parser.parse_args()

    for cl in args.context_lens:
        if cl < 10:
            raise ValueError("context lens <1024 are not supported!")
        if cl > 22:
            warnings.warn("WARNING: context lens >4M may be very slow to generate and have not been extensively tested.")

    generate_dataset(args.dataset, args.context_windows_at_each_len, args.context_lens)


if __name__ == "__main__":
    main()
