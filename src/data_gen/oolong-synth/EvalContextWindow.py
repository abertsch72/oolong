from datasets_loader import EvalDataset
from example_constructor import construct_context
from task_constructors import CountingTasks, UserTasks, TemporalTasks

from typing import Sequence
from time import time

class EvalOutput:
    prompt: str
    question: str
    full_answer: str
    attempted_parse: str
    model: str
    answer_usage: dict
    gold_answer: Sequence[str]

    def __init__(self, prompt, question, model) -> None:
        self.prompt = prompt
        self.question = question
        self.model = model


class EvalContextWindow:
    data: EvalDataset = None
    questions = None

    def __init__(self, data, num_label_classes, context_len, temperature, seed) -> None:
        self.num_label_classes = num_label_classes
        self.num_in_context = data.get_examples_in_context(context_len)

        self.temperature = temperature
        self.data, self.data_stats, self.data_desc_fn = construct_context(
            data, num_label_classes, self.num_in_context, temperature, seed
        )

        self.num_in_context = sum([self.data_stats[dataclass] for dataclass in self.data_stats]) # recalc this in case it's a bit off
        
        before_counting = time()
        self.counting_tasks = CountingTasks(self.data_stats)
        after_counting = time()
        print(f"Counting tasks took {after_counting-before_counting} seconds")
        
        self.user_tasks = UserTasks(self.data_stats, in_context=self.data)
        after_user = time()
        print(f"User tasks took {after_user-after_counting} seconds")

        self.temporal_tasks = TemporalTasks(self.data_stats, in_context=self.data)
        after_temporal = time()
        print(f"Temporal tasks took {after_temporal - after_user} seconds")

    def get_prompt(self, seed=42):
        prompt = self.data_desc_fn(
            intro=True,
            num_examples=self.num_in_context,
            num_labels=self.num_label_classes,
            labels_chosen=list(self.data_stats.keys()),
        )

        prompt_with_labels = f"{prompt}"

        for ex in self.data.sort("x").shuffle(seed=42):
            prompt += f"Date: {ex['formatted_date']} || User: {ex['user_id']} || Instance: {ex['x'].replace('\n', ' ')}\n"
            prompt_with_labels += f"Date: {ex['formatted_date']} || User: {ex['user_id']} || Instance: {ex['x'].replace('\n', ' ')} || Label: {ex['y'].strip()}\n"

        prompt += self.data_desc_fn(
            intro=False,
            num_examples=self.num_in_context,
            num_labels=self.num_label_classes,
            labels_chosen=list(self.data_stats.keys()),
        )

        prompt_with_labels += self.data_desc_fn(
            intro=False,
            num_examples=self.num_in_context,
            num_labels=self.num_label_classes,
            labels_chosen=list(self.data_stats.keys()),
        )

        return prompt, prompt_with_labels

    def evaluate(self, model, seed=None):
        prompt = self.get_prompt(seed)
        outputs = []

        for task in self.counting_tasks.tasks:
            curr_output = EvalOutput(prompt, task.question, model)
            out, usage = model.prompt(prompt, task.question)
            curr_output.full_answer = out
            curr_output.attempted_parse = out.split(":")[-1]
            curr_output.answer_usage = usage
            curr_output.gold_answer = task.answer
            outputs.append(curr_output)

        return outputs
