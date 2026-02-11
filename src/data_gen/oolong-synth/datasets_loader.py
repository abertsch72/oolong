from datasets import load_dataset, Dataset

import json

class EvalDataset:
    x_label = "text"
    y_label = "label"
    intro = ""
    label_map = None
    QUESTION_TOKENS = (
        250  # overestimate, but nothing wrong with leaving a little extra space
    )

    def prep_data(self, rawdata_file):
        with open(rawdata_file, 'r') as f:
            rawdata = [json.loads(ex) for ex in f.readlines()]
        dict_of_lists = {
            "x": [],
            "y": [],
        }
        for ex in rawdata:
            dict_of_lists['x'].append(ex['input'])
            dict_of_lists['y'].append(ex['label'])
        temp = Dataset.from_dict(dict_of_lists)

        return temp

    def get_examples_in_context(self, desired_context_len):
        return int(
            (desired_context_len * 0.95 - self.PROMPT_TOKENS - self.QUESTION_TOKENS)
            // self.PER_EXAMPLE_TOKENS
        )

    def get_general_desc(self, num_examples, specific_text, intro=True):
        if intro:
            start = "The following"
        else:
            start = "Recall: the preceding"
        return f"{start} lines contain {num_examples} {specific_text}\n\nYou will be asked to answer questions about the aggregate label statistics across all {num_examples} examples in this dataset. Do not try to guess, estimate, or approximate the result. Calculate the exact answer given these datapoints.\n\n"


class TREC_Coarse(EvalDataset):
    dataset_name = "CogComp/trec"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 168
    PER_EXAMPLE_TOKENS = 39 

    def __init__(self, split="train"):
        self.label_map = {
            0: "abbreviation",
            1: "entity",
            2: "description and abstract concept",
            3: "human being",
            4: "location",
            5: "numeric value",
        }
        self.label_list = [self.label_map[key] for key in self.label_map]
        self.data = self.prep_data(
            "validated_data/trec_coarse_validated.jsonl"
        )

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        if num_labels == -1:
            num_labels = len(self.label_list)
        if not labels_chosen:
            labels_chosen = self.label_list

        labels_chosen = ", ".join([f"'{label}'" for label in labels_chosen])
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text=f"general-knowledge questions, one per line. Each question has an answer that can be described as one of {num_labels} categories: {labels_chosen}.",
            intro=intro,
        )


class IMDB_Reviews(EvalDataset):
    dataset_name = "stanfordnlp/imdb"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 144
    PER_EXAMPLE_TOKENS = 376

    def __init__(self, split="train"):
        self.label_map = {0: "negative", 1: "positive"}
        self.label_list = [self.label_map[key] for key in self.label_map]
        self.data = self.prep_data(
            "validated_data/imdb_validated.jsonl"
        )

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text="movie reviews, one per line. Each review has a sentiment that can be classified as either 'positive' or 'negative.' There are no neutral reviews.",
            intro=intro,
        )


class AGNews(EvalDataset):
    dataset_name = "fancyzhx/ag_news"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 168
    PER_EXAMPLE_TOKENS = 90

    def __init__(self, split="train"):
        self.label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        self.label_list = [self.label_map[key] for key in self.label_map]
        self.data = self.prep_data(
            "validated_data/agnews_validated.jsonl"
        )

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        if num_labels == -1:
            num_labels = len(self.label_list)
        if not labels_chosen:
            labels_chosen = self.label_list

        labels_chosen = ", ".join([f"'{label}'" for label in labels_chosen])
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text=f"short news articles, one per line. Each article consists of a headline and a lead sentence. The topic of each article can be classified into one of {num_labels} categories: {labels_chosen}.",
            intro=intro,
        )


class HiTZNegation(EvalDataset):
    dataset_name = "HiTZ/This-is-not-a-dataset"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 138
    PER_EXAMPLE_TOKENS = 45

    def __init__(self, split="train"):
        self.label_map = {True: "True", False: "False"}
        self.label_list = ["True", "False"]
        self.data = self.prep_data(
            "validated_data/negation_validated.jsonl"
        )

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        if num_labels == -1:
            num_labels = len(self.label_list)
        if not labels_chosen:
            labels_chosen = self.label_list

        labels_chosen = ", ".join([f"'{label}'" for label in labels_chosen])
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text="sentences, one per line. Each sentence makes a single claim; the claim can be classified as true or false.",
            intro=intro,
        )


class YahooTopics(EvalDataset):
    dataset_name = "community-datasets/yahoo_answers_topics"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 216
    PER_EXAMPLE_TOKENS = 74

    def __init__(self, split="train"):
        self.label_map = {
            0: "Society & Culture",
            1: "Science & Mathematics",
            2: "Health",
            3: "Education & Reference",
            4: "Computers & Internet",
            5: "Sports",
            6: "Business & Finance",
            7: "Entertainment & Music",
            8: "Family & Relationships",
            9: "Politics & Government",
        }
        self.label_list = [self.label_map[key] for key in self.label_map]

        self.data = self.prep_data(
            "validated_data/yahoo_validated.jsonl"
        )

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        if num_labels == -1:
            num_labels = len(self.label_list)
        if not labels_chosen:
            labels_chosen = self.label_list

        labels_chosen = ", ".join([f"'{label}'" for label in labels_chosen])
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text=f"Yahoo Answers questions, one per line. Each line consists of the title and body text of the question. The topic of each question can be classified into one of {num_labels} categories: {labels_chosen}.",
            intro=intro,
        )


class PavlickFormality(EvalDataset):
    dataset_name = "osyvokon/pavlick-formality-scores"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 153
    PER_EXAMPLE_TOKENS = 51

    def __init__(self, split="train"):
        self.label_list = ["formal", "informal"]
        
        self.data = self.prep_data(
            "validated_data/formality_validated.jsonl"
        )

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        if num_labels == -1:
            num_labels = len(self.label_list)
        if not labels_chosen:
            labels_chosen = self.label_list

        labels_chosen = ", ".join([f"'{label}'" for label in labels_chosen])
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text="sentences, one per line. Each sentence can be classified as informal or formal; there are no neutral-style sentences, so each sentence must be given one of these two descriptions.",
            intro=intro,
        )


class SMS_Spam(EvalDataset):
    dataset_name = "ucirvine/sms_spam"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 142
    PER_EXAMPLE_TOKENS = 57

    def __init__(self, split="train"):
        self.label_map = {0: "ham", 1: "spam"}
        self.label_list = [self.label_map[key] for key in self.label_map]
        self.data = self.prep_data(
            "validated_data/spam_validated.jsonl"
        )
        # TODO: classify as formal if score > 0.75, informal if score <-0.75, throw out the intermediate

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        if num_labels == -1:
            num_labels = len(self.label_list)
        if not labels_chosen:
            labels_chosen = self.label_list

        labels_chosen = ", ".join([f"'{label}'" for label in labels_chosen])
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text="text messages, one per line. Each text message can be classified as spam or ham (i.e., not spam).",
            intro=intro,
        )


class MultiNLI(EvalDataset):
    dataset_name = "nyu-mll/multi_nli"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 183
    PER_EXAMPLE_TOKENS = 70

    def __init__(self, split="train"):

        self.label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.label_list = [self.label_map[key] for key in self.label_map]

        self.data = self.prep_data(
            "validated_data/multinli_validated.jsonl"
        )

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        if num_labels == -1:
            num_labels = len(self.label_list)
        if not labels_chosen:
            labels_chosen = self.label_list

        labels_chosen = ", ".join([f"'{label}'" for label in labels_chosen])
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text=f'pairs of sentences, one pair per line. On each line, the two sentences are separated by the marker " -> ". The relationship of the claims in the second sentence to the claims in the first sentence can be classified as one of the following {num_labels} types of relationships: {labels_chosen}.',
            intro=intro,
        )


class MetaphorsBigBench(EvalDataset):
    dataset_name = "tasksource/bigbench"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 173
    PER_EXAMPLE_TOKENS = 51

    def __init__(self, split="train"):
        self.label_list = ["correct", "incorrect"]

        self.data = self.prep_data(
            "validated_data/metaphors_validated.jsonl"
        )

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        if num_labels == -1:
            num_labels = len(self.label_list)
        if not labels_chosen:
            labels_chosen = self.label_list

        labels_chosen = ", ".join([f"'{label}'" for label in labels_chosen])
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text='pairs of sentences, one per line. On each line, the first sentence contains a metaphorical statement and the second sentence (separated by " <--> ") contains a candidate literal interpretation of the statement. The literal interpretation can be classified as either correct or incorrect.',
            intro=intro,
        )


class AppReviews(EvalDataset):
    dataset_name = "sealuzh/app_reviews"

    # computed from an estimate using gpt4 tokenizer and averaging over 1000 examples
    PROMPT_TOKENS = 140
    PER_EXAMPLE_TOKENS = 49

    def __init__(self, split="train"):
        self.orig_x_label = "text"
        self.orig_y_label = "label"

        self.label_list = ["negative", "positive"]

        self.data = self.prep_data(
            "validated_data/app_reviews_validated.jsonl"
        )

    def get_desc(self, num_examples, num_labels=-1, labels_chosen=None, intro=True):
        if num_labels == -1:
            num_labels = len(self.label_list)
        if not labels_chosen:
            labels_chosen = self.label_list

        labels_chosen = ", ".join([f"'{label}'" for label in labels_chosen])
        return self.get_general_desc(
            num_examples=num_examples,
            specific_text="reviews of software applications, one per line. Each review can be classified as positive or negative; there are no neutral reviews.",
            intro=intro,
        )


SUPPORTED_DATASETS = {
    "trec_coarse": TREC_Coarse,                         
    "imdb": IMDB_Reviews,                
    "agnews": AGNews,                                        
    "negation": HiTZNegation, # NEED MORE                           # running llama gen to 8.4k; then will need to redo gpt healing (+5 ex) and run llama healing (+?)
    "yahoo": YahooTopics, 
    "formality": PavlickFormality,
    "spam": SMS_Spam,
    "multinli": MultiNLI, 
    "metaphors": MetaphorsBigBench, 
    "app_reviews": AppReviews, 
}
