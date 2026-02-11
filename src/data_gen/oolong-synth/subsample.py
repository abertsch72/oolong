from datasets import load_dataset
import datasets
import pandas as pd
import pickle
import random

all_short_data = []


long_all = []
prefix = "../../validated_all/"
import os


# more than 2 labels:
    # agnews, trec_coarse, yahoo, multinli 
for file in ['metaphors.pkl', 'negation.pkl', 'formality.pkl', 'spam.pkl', 'imdb.pkl', 'app_reviews.pkl']: #os.listdir(prefix):
    with open(prefix + file, 'rb') as f:
        this_long_data = pickle.load(f)
        assert this_long_data['num_labels'][0] == 2, "need to consider special case for datasets with >2 labels"

    list_of_dicts = []
    for i in range(len(this_long_data['id'])):
        new_dict = {}
        for key, value_list in this_long_data.items():
            new_dict[key] = value_list[i]
            list_of_dicts.append(new_dict)
    for context_len in set(this_long_data['context_len']):
        these_dicts = [i for i in list_of_dicts if i['context_len'] == context_len]
        these_dicts = sorted(these_dicts, key=lambda x: x['context_window_id'])
        if len(these_dicts) < 50:
            print(f"warning for {this_long_data['dataset'][0]} and {context_len}: {len(these_dicts)}") 
        total_added = 0
        if "imdb" in file and context_len == 1024:
            continue
        for context_window_id in set([i['context_window_id'] for i in these_dicts]):
            this_window = [i for i in these_dicts if i['context_window_id'] == context_window_id]
            if context_len == 1024 or (context_len < 5000 and 'imdb' in file): # exception; we'll do 3 windows of size 16-17
                #assert len(this_window) >= 17, f"This window for {file} at {context_len} is short: {len(this_window)}
                remaining_to_add = 50 - total_added
                if remaining_to_add >= len(this_window):
                    all_short_data.extend(this_window)
                    total_added += len(this_window)
                else:
                    all_short_data.extend(this_window[:remaining_to_add])
                    total_added += len(this_window[:remaining_to_add])
            else:
                assert len(this_window) >= 25, f"This window for {file} at {context_len} is short: {len(this_window)}"
                all_short_data.extend(this_window[:25])
                total_added += 25
            if total_added == 50:
                break


len(all_short_data)

all_short_data = [i for i in all_short_data if (i['dataset'] != "imdb" or i['context_len'] != 1024)]

def typename(item):
    return f"{item['task_group']}: {item['task']}"

def get_stats(datasplit):
    tasks = list(set(typename(i) for i in datasplit))
    these_tasks = dict(zip(tasks, [0] * len(tasks)))
    for query in datasplit:
        these_tasks[typename(query)] = these_tasks.get(typename(query), 0) +  1

    for task in these_tasks:
        print(f"{task}: {these_tasks[task]} ({these_tasks[task] / len(datasplit)}) -> should be {these_tasks[task] / len(datasplit) * 50}")

def cut_to_ten(datasplit):
    tasks = list(set(typename(i) for i in datasplit))
    for droptype in tasks:
        new_data = []
        cut_data = []
        for item in datasplit:
            if typename(item) == droptype:
                cut_data.append(item)
            else:
                new_data.append(item)

        random.shuffle(cut_data)
        new_data.extend(cut_data[:14])
        datasplit = new_data
    return new_data

def filter_type(datasplit, droptype, drop_percent):
    
    new_data = []
    for item in datasplit:
        if typename(item) == droptype:
            choice = random.randint(1,100)
            if choice > drop_percent: # keep this example
                new_data.append(item)
        else:
            new_data.append(item)
    return new_data


for multilabel_dataset in ['yahoo.pkl', 'multinli.pkl', 'agnews.pkl', 'trec_coarse.pkl']:
    with open(prefix + multilabel_dataset, 'rb') as f:
        this_long_data = pickle.load(f)

    list_of_dicts = []
    for i in range(len(this_long_data['id'])):
        new_dict = {}
        for key, value_list in this_long_data.items():
            new_dict[key] = value_list[i]
            list_of_dicts.append(new_dict)
    for context_len in set(this_long_data['context_len']):
        these_qs= -1
        trials = 10

        while these_qs < 60:
            these_dicts = [i for i in list_of_dicts if i['context_len'] == context_len]
            #print(len(these_dicts))
            data_to_filter = sorted(these_dicts, key=lambda x: x['context_window_id'])
            
            
            data_to_filter = sorted(data_to_filter, key=lambda x: x['context_window_id'])
            data_to_filter = cut_to_ten(data_to_filter)
            data_to_filter = filter_type(data_to_filter, "user: TASK_TYPE.NUMERIC_ONE_CLASS", 70 - trials)
            data_to_filter = filter_type(data_to_filter, "counting: TASK_TYPE.NUMERIC_ONE_CLASS", 50 - trials)
            data_to_filter = filter_type(data_to_filter, "timeline: TASK_TYPE.NUMERIC_ONE_CLASS", 50 - trials)
            data_to_filter = filter_type(data_to_filter, "user: TASK_TYPE.MOST_FREQ", 10 - trials)
            data_to_filter = filter_type(data_to_filter, "timeline: TASK_TYPE.MOST_FREQ", 50 - trials)
            data_to_filter = filter_type(data_to_filter, "counting: TASK_TYPE.RELATIVE_FREQ", 50 - trials)
            data_to_filter = filter_type(data_to_filter, "timeline: TASK_TYPE.RELATIVE_FREQ", 50 - trials)
            data_to_filter = filter_type(data_to_filter, "user: TASK_TYPE.RELATIVE_FREQ", 50 - trials)
            these_qs = len(data_to_filter)
            trials += 1
            if trials > 20:
                print(trials)
            if these_qs >=60:
                for context_window_id in set([i['context_window_id'] for i in these_dicts]):
                    this_window = [i for i in these_dicts if i['context_window_id'] == context_window_id]
                    #if len(this_window) < 25:
                    #    these_qs=-1


        total_added = 0
        for context_window_id in set([i['context_window_id'] for i in these_dicts]):
            this_window = [i for i in these_dicts if i['context_window_id'] == context_window_id]
            if len(this_window) < 25:
                continue
                assert len(this_window) >= 25, f"This window for {file} at {context_len} is short: {len(this_window)}"
            all_short_data.extend(this_window[:25])
            total_added += 25
            if total_added == 50:
                break

len(all_short_data)

[i['id'] for i in all_short_data][0]

len(set([i['id'] for i in all_short_data]))

while len(set([i['id'] for i in all_short_data])) < 3850:
    ids_seen = []
    for i in all_short_data:
        if i['id'] in ids_seen and int(str(i['id'])[-3:]) < 999:
            i['id'] = i['id'] + 1
            ids_seen.append(i['id'])
        ids_seen.append(i['id'])

# sanity checks
for i in all_short_data:
    if i['dataset'] == 'trec_coarse':
        assert len(str(i['id'])) == 8
    else:
        assert len(str(i['id'])) == 9
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
        assert str(i['id'])[0] == str(supported.index(i['dataset']))
    context_len_to_powers = {
        1024: 10,
        2048: 11,
        4096: 12,
        8192: 13,
        16384: 14,
        32768: 15,
        65536: 16,
        131072: 17,
        262144: 18,
        2**19: 19,
        2**20: 20,
        2**21: 21,
        2**22: 22,
        2**23: 23,

    }

    assert str(i['id'])[-8:-6] == str(context_len_to_powers[i['context_len']]), i['id']
    assert int(str(i['id'])[-3:]) < 900

val_data = [i for i in all_short_data if i['dataset'] in ['trec_coarse', 'spam']]
test_data = [i for i in all_short_data if i['dataset'] not in ['trec_coarse', 'spam']]

from datasets import load_dataset, Dataset

#current_dataset = load_dataset("longreasoningbench/oolong-llamatok-revalidated-synth")

# check number of unique IDs

#current_test = current_dataset['test']
#current_val = current_dataset['validation']

import datasets
from datasets import Dataset, Features

schema = {
    "id": datasets.Value("int64"),
    "context_len": datasets.Value("int64"),
    "dataset": datasets.Value("string"),
    "context_window_text": datasets.Value("string"),
    "context_window_text_with_labels": datasets.Value("string"),
    "question": datasets.Value("string"),
    "task_group": datasets.Value("string"),
    "task": datasets.Value("string"),
    "answer": datasets.Value("string"),
    "answer_type": datasets.Value("string"),
    "input_subset": datasets.Value("string"),
    "num_labels": datasets.Value("int64"),
    "context_window_id": datasets.Value("int64"),
}

from datasets import Features



dict_of_lists_test = {}
remainder = len(test_data)
for d in test_data:
    for key, value in d.items():
        if key not in dict_of_lists_test:
            dict_of_lists_test[key] = []
        dict_of_lists_test[key].append(value)
    remainder -= 1
    #if remainder % 100 == 0:
    #    print(f"{remainder} left")



new_test_hf= Dataset.from_dict(dict_of_lists_test,features=Features(schema))
new_test_hf

dict_of_lists_val = {}
for d in val_data:
    for key, value in d.items():
        if key not in dict_of_lists_val:
            dict_of_lists_val[key] = []
        dict_of_lists_val[key].append(value)



new_val_hf= Dataset.from_dict(dict_of_lists_val,features=Features(schema))
new_val_hf



from datasets import DatasetDict

final_new_data = DatasetDict({
    "validation": new_val_hf,
    "test": new_test_hf
})

final_new_data.save_to_disk("oolong-synth-datagen/subsampled_oolong_synth")



