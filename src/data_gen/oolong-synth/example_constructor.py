import random
import math


import torch
from datasets import concatenate_datasets
import numpy as np
from utils import sample_random_dates, generate_skewed_user_ids


def construct_context(
    dataobj, num_label_classes, num_in_context, temperature=1, seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    labels = dataobj.label_list
    dataset = dataobj.data
    if num_label_classes != -1:
        labels = random.sample(labels, num_label_classes)
        dataset = dataset.filter(lambda example: example["y"] in labels)

    random.shuffle(labels)
    distribution = torch.nn.functional.softmax(
        torch.tensor(
            [random.randint(1, 30) ** (1 / temperature) for i in range(len(labels))],
            dtype=float,
        )
    )
    counts = (distribution * num_in_context).round()
    counts[0] = counts[0] - (
        sum(counts) - num_in_context
    )  # corrects for rounding, now the number of examples will be exactly correct

    distribution = (
        counts / num_in_context
    )  # corrects the distribution after our rounding adjustment
    true_counts = {}
    sampled_datasets = []
    for label_name, num_samples in zip(labels, counts):
        # Filter dataset to only include examples with this label
        label_filtered = dataset.filter(lambda example: example["y"] == label_name)
        num_samples = int(num_samples.item())
        # Check if we have enough examples
        total_available = len(label_filtered)
        if total_available < num_samples:
            print(
                f"Warning: Requested {num_samples} samples for label '{label_name}', but only {total_available} are available. Using duplicate examples."
            )
            copies_needed = math.ceil(num_samples / total_available)
            label_filtered = concatenate_datasets([label_filtered] * copies_needed)
            total_available = len(label_filtered)

        # Select random indices for sampling
        if num_samples > 0:
            # Generate random indices without replacement
            if total_available > 0:
                indices = list(range(total_available))
                random.shuffle(indices)
                indices = indices[:num_samples]
                sampled_subset = label_filtered.select(indices)
                sampled_datasets.append(sampled_subset)
        true_counts[label_name] = num_samples

    final_data = concatenate_datasets(sampled_datasets)

    final_data = final_data.shuffle()
    num_in_context = len(final_data)  # update in case we resampled
    user_ids = generate_skewed_user_ids(num_in_context)
    dateobjs, formatted_dates = sample_random_dates(num_in_context)
    final_data = final_data.add_column("user_id", user_ids)
    final_data = final_data.add_column("dateobj", dateobjs)
    final_data = final_data.add_column("formatted_date", formatted_dates)

    return final_data, true_counts, dataobj.get_desc


def most_common_label(context_data, true_counts):
    true_answer = max(true_counts, key=true_counts.get)
    label_list = ", ".join(list(true_counts.keys()))
    question = f"\n\nIn the above data, which of the labels is the most common? Give your final answer in the form 'Label: answer' where answer is one of the labels: {label_list}.\n\n"
    return [question], [true_answer]
