"""Generate question-answer pairs for Critical Role dataset."""

import hashlib
import json
import random
import re
import sys
import uuid
from pathlib import Path

import fire
import pandas as pd

# import tiktoken
from loguru import logger

# from transformers import AutoTokenizer

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
    level="INFO",
)

SEED = 31

QUESTIONS = {
    "rolls": [
        "Total number of rolls in this episode?",
        "Total number of rolls by the character {character_name} in this episode?",
        "Total number of rolls by the player {player_name} in this episode?",
        "Total number of rolls of type {roll_type} in this episode?",
        "Number of rolls of natural value {roll_value} in this episode?",
        "In this episode, what percentage of rolls were of value {roll_value}? round to the nearest integer.",
        "What is the most common roll type in this episode? Return a comma separated list.",
        "What is the least common roll type in this episode? Only include types with more than one roll. Return a comma separated list.",
        "What is the most common natural roll value in this episode? Return a comma separated list.",
        "What is the least common natural roll value in this episode? Only include values with more than one roll. Return a comma separated list.",
        "What is the count of Crits? (natural rolls of value 1 or 20)?",
        "What is the count of Nat20s (natural rolls of value 20)?",
        "What is the count of Nat1s (natural rolls of value 1)?",
    ],
    "spells": [
        "How many spells were cast during this episode?",
        "How many spells were cast by the character {character_name} in this episode?",
        "How many spells were cast by the player {player_name} in this episode?",
        "How many {spell_type} spells were cast during this episode?",
        "What are the first {count} spells cast in this episode? Return a comma separated list.",
        "What are the last {count} spells cast in this episode? Return a comma separated list.",
        "Return a comma separated list and retain the order of spells as they appear in the episode.",
        "What is the last spell cast by each character in this episode? Return a comma separated list and retain the order of spells as they appear in the episode.",
        "How many characters cast {spell_name} spell in this episode?",
        "What is the most common spell in this episode? Return a comma separated list.",
        "What is the least common spell in this episode? Only include spells that were cast at least once. Return a comma separated list.",
        "Which spells were cast by more than one character in this episode? Return a comma separated list.",
        "What is the total number of cantrip spells cast in this episode?",
        "In this episode, how many times was a spell cast at a level higher than its base level?",
        "In this episode, which spells were cast at a level higher than their base level? Return a comma separated list of unique spells.",
    ],
    "hdywtdt": [
        "How many times does the Dungeon Master ask 'How do you want to do this?' in this episode?",
    ],
    "kills": [
        "Total number of kills in this episode?",
        "What are the total number of kills by the character {character_name} in this episode?",
    ],
    "damage": [
        "What is the total damage dealt during this episode?",
        "What is the total damage dealt by the character {character_name} in this episode?",
        "What is the total damage taken during this episode?",
        "What is the total damage taken by the character {character_name} in this episode?",
    ],
    "multidoc_rolls": [
        "What is the cummulative total of rolls by the end of episode {episode_index}? Count the number of rolls and not the values of the rolls.",
        "What is the cummulative total of rolls by the character {character_name} at the end of episode {episode_index}? Count the number of rolls and not the values of the rolls.",
        "Total number of rolls across all the episodes?",
        "Total number of rolls by the character {character_name} across all episodes?",
        "Total number of rolls by the player {player_name} across all episodes?",
        "Total number of rolls of type {roll_type} across all episodes?",
        "Number of rolls of natural value {roll_value} across all episodes?",
        "Across all episodes, what percentage of rolls were of value {roll_value}? round to the nearest integer.",
        "What is the most common roll type across all episodes? Return a comma separated list.",
        "What is the least common roll type across all episodes? Only include types with more than one roll. Return a comma separated list.",
        "What is the most common natural roll value across all episodes? Return a comma separated list.",
        "What is the least common natural roll value across all episodes? Only include values with more than one roll. Return a comma separated list.",
        "What is the total count of Crits across all episodes? (natural rolls of value 1 or 20)?",
        "What is the total count of Nat20s across all episodes? (natural rolls of value 20)?",
        "What is the total count of Nat1s across all episodes? (natural rolls of value 1)?",
    ],
    "multidoc_spells": [
        "What is the cummulative total of spells cast by the end of episode {episode_index}?",
        "What is the first spell cast in the episode {episode_index}?",
        "What is the second spell cast in the episode {episode_index}?",
        "What is the third spell cast in the episode {episode_index}?",
        "List the first spell cast in each episode? Return a comma separated list.",
        "List the last spell cast in each episode? Return a comma separated list.",
        "List the first spell cast by the character {character_name} in each episode? Return a comma separated list.",
        "List the last spell cast by the character {character_name} in each episode? Return a comma separated list.",
        "How many spells were cast across all episodes?",
        "How many spells were cast by the character {character_name} across all episodes?",
        "How many spells were cast by the player {player_name} across all episodes?",
        "How many {spell_type} spells were cast across all episodes?",
        "How many characters cast {spell_name} spell across all episodes?",
        "What is the most common spell across all episodes? Return a comma separated list.",
        "What is the least common spell across all episodes? Only include spells that were cast at least once. Return a comma separated list.",
        "What is the total number of cantrip spells cast across all episodes?",
        "Across all episodes, how many times was a spell cast at a level higher than its base level?",
        "Across all episodes, which spells were cast at a level higher than their base level? Return a comma separated list of unique spells.",
    ],
}


def generate_unique_uuid(text: str) -> str:
    hash_bytes = hashlib.md5(text.encode()).digest()

    # Create UUID from hash
    return str(uuid.UUID(bytes=hash_bytes))


def load_p2c(file_path: str) -> dict:
    """Load mapping between player names and character names."""
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["Player"].notna()]
    return dict(zip(df["Player"], df["Character"]))


def rolls(df: pd.DataFrame, p2c: dict, toy_dataset: bool = False) -> tuple:
    """Generate question, answer pairs for rolls."""
    rnd = random.Random(SEED)
    roll_values = [str(x) for x in range(1, 21)]
    # filter empty roll types
    roll_types = [
        roll_type
        for roll_type in df["Type of Roll"].unique()
        if pd.notna(roll_type)
        and roll_type.strip() != ""
        and roll_type.lower() not in ["other", "unknown"]
    ]
    character_names = [
        name
        for name in df["Character"].unique()
        if not pd.isna(name) and name.lower() not in ["others", "unknown"]
    ]

    character_names = rnd.sample(list(character_names), min(2, len(character_names)))
    roll_values = rnd.sample(roll_values, min(2, len(roll_values)))
    roll_types = rnd.sample(roll_types, min(2, len(roll_types)))

    # player names for sampled character names
    player_names = [p for p in p2c if p2c[p] in character_names and p2c[p] != "DM"]

    logger.debug("roll values: {}", roll_values)
    logger.debug("roll types: {}", roll_types)
    logger.debug("character names: {}", character_names)
    logger.debug("p2c: {}", p2c)

    questions_answers = [
        ("Total number of rolls in this episode?", df.shape[0]),
        *[
            (
                f"Total number of rolls by the character {character} in this episode?",
                df[df["Character"] == character].shape[0],
            )
            for character in character_names
        ],
        *[
            (
                f"Total number of rolls by the player {player} in this episode?",
                df[df["Character"] == p2c[player]].shape[0],
            )
            for player in player_names
        ],
        *[
            (
                f"Total number of rolls of type {roll_type} in this episode?",
                df[df["Type of Roll"] == roll_type].shape[0],
            )
            for roll_type in roll_types
        ],
        *[
            (
                f"Number of rolls of natural value {roll_value} in this episode?",
                df[df["Natural Value"] == roll_value].shape[0],
            )
            for roll_value in roll_values
        ],
        *[
            (
                f"In this episode, what percentage of rolls were of value {roll_value}? round to the nearest integer.",
                round(
                    df[df["Natural Value"] == roll_value].shape[0] / df.shape[0] * 100
                ),
            )
            for roll_value in roll_values
        ],
        (
            "What is the most common roll type in this episode? Return a comma separated list.",
            ", ".join(df["Type of Roll"].mode().tolist()),
        ),
        (
            "What is the least common roll type in this episode? Only include types with more than one roll. Return a comma separated list.",
            ", ".join(
                df["Type of Roll"]
                .value_counts()[
                    df["Type of Roll"].value_counts()
                    == df["Type of Roll"].value_counts().min()
                ]
                .index.tolist()
            ),
        ),
        (
            "What is the most common natural roll value in this episode? Return a comma separated list.",
            ", ".join(df["Natural Value"].mode().tolist()),
        ),
        (
            "What is the least common natural roll value in this episode? Only include values with more than one roll. Return a comma separated list.",
            ", ".join(
                df["Natural Value"]
                .value_counts()[
                    df["Natural Value"].value_counts()
                    == df["Natural Value"].value_counts().min()
                ]
                .index.tolist()
            ),
        ),
        (
            "What is the count of Crits? (natural rolls of value 1 or 20)?",
            df[df["Natural Value"].isin(["1", "20"])].shape[0],
        ),
        (
            "What is the count of Nat20s (natural rolls of value 20)?",
            df[df["Natural Value"] == "20"].shape[0],
        ),
        (
            "What is the count of Nat1s (natural rolls of value 1)?",
            df[df["Natural Value"] == "1"].shape[0],
        ),
    ]

    return zip(*questions_answers)


def spells(df: pd.DataFrame, p2c: dict, toy_dataset: bool = False) -> tuple:
    """Generate question, answer pairs for spells."""
    rnd = random.Random(SEED)
    spell_types = [
        spell
        for spell in df["Spell"].unique()
        if pd.notna(spell) and spell.lower() not in ["unknown", "other"]
    ]
    character_names = [
        name
        for name in df["Character"].unique()
        if not pd.isna(name) and name.lower() not in ["others", "unknown"]
    ]

    # sample 2 character names, player names, and spell types
    character_names = rnd.sample(list(character_names), min(2, len(character_names)))
    spell_types = rnd.sample(list(spell_types), min(2, len(spell_types)))

    # player names for sampled character names
    player_names = [p for p in p2c if p2c[p] in character_names and p2c[p] != "DM"]

    questions_answers = [
        ("How many spells were cast during this episode?", df.shape[0]),
        *[
            (
                f"How many spells were cast by the character {character} in this episode?",
                df[df["Character"] == character].shape[0],
            )
            for character in character_names
        ],
        *[
            (
                f"How many spells were cast by the player {player} in this episode?",
                df[df["Character"] == p2c[player]].shape[0],
            )
            for player in player_names
        ],
        *[
            (
                f"How many {spell_type} spells were cast during this episode?",
                df[df["Spell"] == spell_type].shape[0],
            )
            for spell_type in spell_types
        ],
        ("What is the first spell cast in this episode?", df["Spell"].iloc[0]),
        (
            "What are the first two spells cast in this episode?",
            ", ".join(df.head(2)["Spell"].tolist()),
        ),
        (
            "What are the first three spells cast in this episode?",
            ", ".join(df.head(3)["Spell"].tolist()),
        ),
        ("What is the last spell cast in this episode?", df["Spell"].iloc[-1]),
        (
            "What are the last two spells cast in this episode?",
            ", ".join(df.tail(2)["Spell"].tolist()),
        ),
        (
            "What are the last three spells cast in this episode?",
            ", ".join(df.tail(3)["Spell"].tolist()),
        ),
        [
            "What is the first spell cast by each character in this episode? Return a comma separated list and retain the order of spells as they appear in the episode.",
            ", ".join(df.groupby("Character")["Spell"].first().tolist()),
        ],
        [
            "What is the last spell cast by each character in this episode? Return a comma separated list and retain the order of spells as they appear in the episode.",
            ", ".join(df.groupby("Character")["Spell"].last().tolist()),
        ],
        *[
            (
                f"How many characters cast {spell_name} spell in this episode?",
                df[df["Spell"] == spell_name]["Character"].nunique(),
            )
            for spell_name in spell_types
        ],
        [
            "What is the most common spell in this episode? Return a comma separated list.",
            ", ".join(df["Spell"].mode().tolist()),
        ],
        [
            "What is the least common spell in this episode? Only include spells that were cast at least once. Return a comma separated list.",
            ", ".join(
                df["Spell"]
                .value_counts()[
                    df["Spell"].value_counts() == df["Spell"].value_counts().min()
                ]
                .index.tolist()
            ),
        ],
        [
            "Which spells were cast by more than one character in this episode? Return a comma separated list.",
            ", ".join(
                spell_name
                for spell_name in df["Spell"].unique()
                if df[df["Spell"] == spell_name]["Character"].nunique() > 1
            ),
        ],
        [
            "What is the total number of cantrip spells cast in this episode?",
            df[df["Base Lvl"] == "Cantrip"].shape[0],
        ],
        [
            "In this episode, how many times was a spell cast at a level higher than its base level?",
            df[df["Cast At"] > df["Base Lvl"]].shape[0],
        ],
        [
            "In this episode, which spells were cast at a level higher than their base level? Return a comma separated list of unique spells.",
            ", ".join(df[df["Cast At"] > df["Base Lvl"]]["Spell"].unique().tolist()),
        ],
    ]

    return zip(*questions_answers)


def kills_questions(df: pd.DataFrame, character2player: dict) -> tuple:
    """Generate question, answer pairs for kills."""
    character_names = df["Character"].unique()
    # kill counts for each roll in the Kills column
    # filter rows that where Kills is an integer
    df = df[df["# Kills"].apply(lambda x: str(x).isdigit())]
    df["# Kills"] = df["# Kills"].astype(int)
    questions_answers = [
        ("Total number of kills in this episode?", int(df["# Kills"].sum())),
        *[
            (
                f"What are the total number of kills by character {character} in this episode?",
                int(df[df["Character"] == character]["# Kills"].sum()),
            )
            for character in character_names
        ],
    ]

    return zip(*questions_answers)


def multidoc_rolls(
    df: pd.DataFrame,
    episodes: list[int],
    p2c: dict,
    sample_size_probe: int,
    toy_dataset: bool = False,
) -> tuple:
    """Generate question, answer pairs for rolls in multi-episode setup."""
    # episodes contains a list of episode numbers
    # we want to get the cumulative total of rolls by the end of each episode (first, second, etc.,)
    roll_counts = [df[df["Episode"] == episode].shape[0] for episode in episodes]
    cummulative_roll_counts = list(pd.Series(roll_counts).cumsum())
    character_names = [
        name
        for name in df["Character"].unique()
        if not pd.isna(name) and name.lower() not in ["others", "unknown"]
    ]
    roll_values = [str(x) for x in range(1, 21)]
    roll_types = [
        roll_type
        for roll_type in df["Type of Roll"].unique()
        if pd.notna(roll_type)
        and roll_type.strip() != ""
        and roll_type.lower() not in ["other", "unknown"]
    ]
    rnd = random.Random(SEED)

    character_names = rnd.sample(list(character_names), min(2, len(character_names)))
    roll_values = rnd.sample(roll_values, min(2, len(roll_values)))
    roll_types = rnd.sample(roll_types, min(2, len(roll_types)))

    # player names for sampled character names
    player_names = [p for p in p2c if p2c[p] in character_names and p2c[p] != "DM"]

    # sample episodes for QA pairs
    sampled_indices = sorted(
        rnd.sample(range(len(episodes)), min(sample_size_probe, len(episodes)))
    )
    questions_answers = [
        (
            f"What is the cummulative total of rolls by the end of episode {episode_index + 1}? Count the number of rolls and not the values of the rolls.",
            cummulative_roll_counts[episode_index],
        )
        for episode_index in sampled_indices
    ]

    for character in character_names:
        if pd.isna(character) or character == "Others":
            continue

        roll_counts = [
            df[(df["Episode"] == episode) & (df["Character"] == character)].shape[0]
            for episode in episodes
        ]
        cummulative_roll_counts = list(pd.Series(roll_counts).cumsum())

        # sample episodes for QA pairs
        sampled_indices = sorted(
            rnd.sample(range(len(episodes)), min(sample_size_probe, len(episodes)))
        )
        questions_answers += [
            (
                f"What is the cummulative total of rolls by the character {character} at the end of episode {episode_index + 1}? Count the number of rolls and not the values of the rolls.",
                cummulative_roll_counts[episode_index],
            )
            for episode_index in sampled_indices
        ]

    questions_answers += [
        ("Total number of rolls across all the episodes?", df.shape[0]),
        *[
            (
                f"Total number of rolls by the character {character} across all episodes?",
                df[df["Character"] == character].shape[0],
            )
            for character in character_names
        ],
        *[
            (
                f"Total number of rolls by the player {player} across all episodes?",
                df[df["Character"] == p2c[player]].shape[0],
            )
            for player in player_names
        ],
    ]
    questions_answers += [
        (
            f"Total number of rolls of type {roll_type} across all episodes?",
            df[df["Type of Roll"] == roll_type].shape[0],
        )
        for roll_type in roll_types
    ]
    questions_answers += [
        (
            f"Number of rolls of natural value {roll_value} across all episodes?",
            df[df["Natural Value"] == roll_value].shape[0],
        )
        for roll_value in roll_values
    ]
    questions_answers += [
        (
            f"Across all episodes, what percentage of rolls were of value {roll_value}? round to the nearest integer.",
            round(df[df["Natural Value"] == roll_value].shape[0] / df.shape[0] * 100),
        )
        for roll_value in roll_values
    ]
    questions_answers += [
        (
            "What is the most common roll type across all episodes? Return a comma separated list.",
            ", ".join(df["Type of Roll"].mode().tolist()),
        ),
        (
            "What is the least common roll type across all episodes? Only include types with more than one roll. Return a comma separated list.",
            ", ".join(
                df["Type of Roll"]
                .value_counts()[
                    df["Type of Roll"].value_counts()
                    == df["Type of Roll"].value_counts().min()
                ]
                .index.tolist()
            ),
        ),
        (
            "What is the most common natural roll value across all episodes? Return a comma separated list.",
            ", ".join(df["Natural Value"].mode().tolist()),
        ),
        (
            "What is the least common natural roll value across all episodes? Only include values with more than one roll. Return a comma separated list.",
            ", ".join(
                df["Natural Value"]
                .value_counts()[
                    df["Natural Value"].value_counts()
                    == df["Natural Value"].value_counts().min()
                ]
                .index.tolist()
            ),
        ),
        (
            "What is the total count of Crits across all episodes? (natural rolls of value 1 or 20)?",
            df[df["Natural Value"].isin(["1", "20"])].shape[0],
        ),
        (
            "What is the total count of Nat20s across all episodes? (natural rolls of value 20)?",
            df[df["Natural Value"] == "20"].shape[0],
        ),
        (
            "What is the total count of Nat1s across all episodes? (natural rolls of value 1)?",
            df[df["Natural Value"] == "1"].shape[0],
        ),
    ]

    return zip(*questions_answers)


def multidoc_spells(
    df: pd.DataFrame,
    episodes: list[int],
    p2c: dict,
    sample_size_probe: int,
    toy_dataset: bool = False,
) -> tuple:
    """Generate question, answer pairs for spells in multi-episode setup."""
    spell_counts = [df[df["Episode"] == episode].shape[0] for episode in episodes]
    cummulative_spell_counts = list(pd.Series(spell_counts).cumsum())
    character_names = [
        name
        for name in df["Character"].unique()
        if not pd.isna(name) and name.lower() not in ["others", "unknown"]
    ]
    spell_types = [
        spell
        for spell in df["Spell"].unique()
        if pd.notna(spell) and spell.lower() not in ["unknown", "other"]
    ]
    rnd = random.Random(SEED)

    character_names = rnd.sample(list(character_names), min(2, len(character_names)))
    spell_types = rnd.sample(list(spell_types), min(2, len(spell_types)))

    # player names for sampled character names
    player_names = [p for p in p2c if p2c[p] in character_names and p2c[p] != "DM"]

    sampled_indices = sorted(
        rnd.sample(range(len(episodes)), min(sample_size_probe, len(episodes)))
    )
    questions_answers = [
        (
            f"What is the cummulative total of spells cast by the end of episode {episode_index + 1}?",
            cummulative_spell_counts[episode_index],
        )
        for episode_index in sampled_indices
    ]

    # first spell in a given episode
    sampled_indices = sorted(
        rnd.sample(range(len(episodes)), min(sample_size_probe, len(episodes)))
    )
    questions_answers += [
        (
            f"What is the first spell cast in the episode {episode_index + 1}?",
            df[df["Episode"] == episodes[episode_index]]["Spell"].iloc[0],
        )
        for episode_index in sampled_indices
    ]

    # second spell in a given episode
    sampled_indices = sorted(
        rnd.sample(range(len(episodes)), min(sample_size_probe, len(episodes)))
    )
    questions_answers += [
        (
            f"What is the second spell cast in the episode {episode_index + 1}?",
            df[df["Episode"] == episodes[episode_index]]["Spell"].iloc[1],
        )
        for episode_index in sampled_indices
    ]

    # third spell in a given episode
    sampled_indices = sorted(
        rnd.sample(range(len(episodes)), min(sample_size_probe, len(episodes)))
    )
    for episode_index in sampled_indices:
        if df[df["Episode"] == episodes[episode_index]].shape[0] < 3:
            continue
        questions_answers += [
            (
                f"What is the third spell cast in the episode {episode_index + 1}?",
                df[df["Episode"] == episodes[episode_index]]["Spell"].iloc[2],
            )
        ]
    questions_answers += [
        (
            "List the first spell cast in each episode? Return a comma separated list.",
            ", ".join(df.groupby("Episode")["Spell"].first().tolist()),
        ),
        (
            "List the last spell cast in each episode? Return a comma separated list.",
            ", ".join(df.groupby("Episode")["Spell"].last().tolist()),
        ),
    ]
    # first spell cast by a given character in each episode
    questions_answers += [
        (
            f"List the first spell cast by the character {character} in each episode? Return a comma separated list.",
            ", ".join(
                df[df["Character"] == character]
                .groupby("Episode")["Spell"]
                .first()
                .tolist()
            ),
        )
        for character in character_names
    ]
    # last spell cast by a given character in each episode
    questions_answers += [
        (
            f"List the last spell cast by the character {character} in each episode? Return a comma separated list.",
            ", ".join(
                df[df["Character"] == character]
                .groupby("Episode")["Spell"]
                .last()
                .tolist()
            ),
        )
        for character in character_names
    ]
    questions_answers += [
        ("How many spells were cast across all episodes?", df.shape[0]),
        *[
            (
                f"How many spells were cast by the character {character} across all episodes?",
                df[df["Character"] == character].shape[0],
            )
            for character in character_names
        ],
        *[
            (
                f"How many spells were cast by the player {player} across all episodes?",
                df[df["Character"] == p2c[player]].shape[0],
            )
            for player in player_names
        ],
    ]
    questions_answers += [
        (
            f"How many {spell_type} spells were cast across all episodes?",
            df[df["Spell"] == spell_type].shape[0],
        )
        for spell_type in spell_types
    ]
    questions_answers += [
        (
            f"How many characters cast {spell_name} spell across all episodes?",
            df[df["Spell"] == spell_name]["Character"].nunique(),
        )
        for spell_name in spell_types
    ]
    questions_answers += [
        (
            "What is the most common spell across all episodes? Return a comma separated list.",
            ", ".join(df["Spell"].mode().tolist()),
        ),
        (
            "What is the least common spell across all episodes? Only include spells that were cast at least once. Return a comma separated list.",
            ", ".join(
                df["Spell"]
                .value_counts()[
                    df["Spell"].value_counts() == df["Spell"].value_counts().min()
                ]
                .index.tolist()
            ),
        ),
        (
            "What is the total number of cantrip spells cast across all episodes?",
            df[df["Base Lvl"] == "Cantrip"].shape[0],
        ),
        (
            "Across all episodes, how many times was a spell cast at a level higher than its base level?",
            df[df["Cast At"] > df["Base Lvl"]].shape[0],
        ),
        (
            "Across all episodes, which spells were cast at a level higher than their base level? Return a comma separated list of unique spells.",
            ", ".join(df[df["Cast At"] > df["Base Lvl"]]["Spell"].unique().tolist()),
        ),
    ]
    return zip(*questions_answers)


def write_qa_pairs(
    file_path: str, questions: list, answers: list, episodes: list
) -> None:
    """Write question-answer pairs to a file."""
    with open(file_path, "w") as f:
        for question, answer in zip(questions, answers):
            question = str(question).replace('"', '\\"')
            if isinstance(answer, str):
                answer = str(answer).replace('"', '\\"')
            if isinstance(answer, list):
                answer = [str(x).replace('"', '\\"') for x in answer]
            f.write(json.dumps({"question": question, "answer": answer}) + "\n")
    logger.info("writing QA pairs to {}", file_path)


def clean_text(text: str | list[str]) -> str | list[str]:
    if isinstance(text, int) or isinstance(text, float):
        return str(text)
    if isinstance(text, str):
        return str(text).replace('"', '\\"')
    if isinstance(text, list):
        return [str(x).replace('"', '\\"') for x in text]
    return text


def load_transcripts(dir_path: Path) -> dict:
    transcripts = {}
    for file_path in dir_path.glob("C*E*.txt"):
        # C1E020.txt
        match = re.search(r"E(\d+)", file_path.name)
        if match:
            episode_number = int(match.group(1))
            with file_path.open("r") as f:
                transcripts[episode_number] = f.read()
    return transcripts


def filter_qa_pairs(questions: list, answers: list):
    filtered_questions = []
    filtered_answers = []
    for idx in range(len(questions)):
        question = clean_text(questions[idx])
        answer = clean_text(answers[idx])
        if (
            answer is None
            or answer == ""
            or answer.lower() in ["nan", "none", "unknown"]
        ):
            continue
        filtered_questions.append(question)
        filtered_answers.append(answer)
    return filtered_questions, filtered_answers


def main(
    stats_dir: str,
    transcripts_dir: str,
    campaign: str,
    split: str,
    output_dir: str,
    toy_dataset: bool = False,
) -> None:
    """Load tsv files for statistics, write QA pairs."""
    stats_dir = Path(stats_dir)
    output_dir = Path(output_dir)
    transcripts_dir = Path(transcripts_dir)
    stats_dir = stats_dir / campaign
    transcripts_dir = transcripts_dir / campaign
    if toy_dataset:
        output_dir = output_dir / "toy_dnd"
    else:
        output_dir = output_dir / "dnd"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load player names
    p2c = load_p2c(stats_dir / "characters.tsv")

    rolls_df = pd.read_csv(stats_dir / "rolls.tsv", sep="\t")
    spells_df = pd.read_csv(stats_dir / "spells.tsv", sep="\t")

    all_episodes = sorted(spells_df["Episode"].unique())

    transcripts = load_transcripts(transcripts_dir)
    for episode in all_episodes:
        if episode not in transcripts:
            logger.warning("transcript for episode {} not found", episode)
    all_episodes = [episode for episode in all_episodes if episode in transcripts]

    # llama_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    # openai_tok = tiktoken.get_encoding("cl100k_base")
    SINGLEDOC_PROMPT = (
        "The following lines contains a single episode transcript of a Dungeons and Dragons game played by a group of players. "
        "The episode transcript is delimited by [START OF EPISODE] and [END OF EPISODE]. "
        "The transcript is followed by a question about the game statistics. "
        "Answer the question based on the transcript. "
        "Do not try to guess, estimate or approximate the result. "
        "Do not ask the user for clarification or follow-up. "
        "Do step-by-step reasoning if needed. "
        "Return the final answer in \\boxed{{}}. "
    )

    MULTIDOC_PROMPT = (
        "The following lines contains multiple episode transcripts of a Dungeons and Dragons game played by a group of players. "
        "Each episode transcript is delimited by [START OF EPISODE] and [END OF EPISODE]. "
        "The transcripts are followed by a question about the game statistics. "
        "Answer the question based on the transcripts. "
        "Do not try to guess, estimate or approximate the result. "
        "Do not ask the user for clarification or follow-up. "
        "Do step-by-step reasoning if needed. "
        "Return the final answer in \\boxed{{}}. "
    )
    if toy_dataset:
        # number of episode set samples for a given context window length
        sample_size_per_window = 1
        # number of episodes to probe with QA pairs for a given set of episodes, multi-episode only
        sample_size_probe = 1
    else:
        sample_size_per_window = 10
        sample_size_probe = 3

    # create prompt to include with player to character mapping
    p2c_lines = [
        f"{player} plays the character {character}."
        for player, character in p2c.items()
    ]
    p2c_prompt = (
        "The following lines contain the mapping between player names and character names.\n"
        + "\n".join(p2c_lines)
        + "\n\nUse this mapping when answering the questions below.\n\n"
    )
    SINGLEDOC_PROMPT += "\n\n" + p2c_prompt
    MULTIDOC_PROMPT += "\n\n" + p2c_prompt

    # generate QA pairs for each episode, write to jsonl file
    output_file_path = output_dir / f"{split}.jsonl"
    logger.info("writing singledoc QA pairs to {}", output_file_path)
    with output_file_path.open("w") as f:
        rnd = random.Random(SEED)
        sampled_episodes = sorted(rnd.sample(all_episodes, sample_size_per_window))
        for episode in sampled_episodes:
            # QA pairs for rolls
            questions, answers, types = [], [], []
            questions_rolls, answers_rolls = rolls(
                rolls_df[rolls_df["Episode"] == episode], p2c, toy_dataset=toy_dataset
            )
            questions.extend(questions_rolls)
            answers.extend(answers_rolls)
            types.extend(["singledoc_rolls"] * len(questions_rolls))

            # QA pairs for spells
            questions_spells, answers_spells = spells(
                spells_df[spells_df["Episode"] == episode], p2c, toy_dataset=toy_dataset
            )
            questions.extend(questions_spells)
            answers.extend(answers_spells)
            types.extend(["singledoc_spells"] * len(questions_spells))
            # NOTE: Kills data might be inaccurate, leaving it out for now
            # write questions and answers to a single JSONL file
            # filter out QA pairs with empty answers
            questions, answers = filter_qa_pairs(questions, answers)
            for idx in range(len(questions)):
                assert questions[idx] is not None and answers[idx] is not None
                episodes = [int(episode)]
                context_window_text = SINGLEDOC_PROMPT + "\n\n".join(
                    [
                        f"[START OF EPISODE]\n{transcripts[episode]}\n[END OF EPISODE]"
                        for episode in episodes
                    ]
                )
                output_dict = {
                    "id": generate_unique_uuid(
                        campaign + questions[idx] + "_".join(map(str, episodes))
                    ),
                    "context_window_id": generate_unique_uuid(
                        campaign + "_".join(map(str, episodes))
                    ),
                    "context_window_text": context_window_text,
                    "question": questions[idx],
                    "answer": answers[idx],
                    "question_type": types[idx],
                    "episodes": episodes,
                    "campaign": campaign,
                }
                f.write(json.dumps(output_dict) + "\n")

    logger.info("writing multidoc QA pairs to {}", output_file_path)
    context_windows = [2, 3, 4, 5, 6, 7, 8, 10, 16, 24]
    with output_file_path.open("a") as f:
        for num_episodes in context_windows:
            rnd = random.Random(SEED)
            for sample_idx in range(sample_size_per_window):
                context_window_episodes = rnd.sample(all_episodes, num_episodes)
                context_window_episodes = sorted(context_window_episodes)
                questions, answers, types = [], [], []
                # QA pairs for rolls
                questions_rolls, answers_rolls = multidoc_rolls(
                    rolls_df[rolls_df["Episode"].isin(context_window_episodes)],
                    context_window_episodes,
                    p2c=p2c,
                    sample_size_probe=sample_size_probe,
                    toy_dataset=toy_dataset,
                )
                questions.extend(questions_rolls)
                answers.extend(answers_rolls)
                types.extend(["multidoc_rolls"] * len(questions_rolls))
                # QA pairs for spells
                questions_spells, answers_spells = multidoc_spells(
                    spells_df[spells_df["Episode"].isin(context_window_episodes)],
                    context_window_episodes,
                    p2c=p2c,
                    sample_size_probe=sample_size_probe,
                    toy_dataset=toy_dataset,
                )
                questions.extend(questions_spells)
                answers.extend(answers_spells)
                types.extend(["multidoc_spells"] * len(questions_spells))
                # write questions and answers to a single JSONL file
                assert len(questions) == len(answers)
                # filter out QA pairs with empty answers
                questions, answers = filter_qa_pairs(questions, answers)
                for idx in range(len(questions)):
                    assert questions[idx] is not None and answers[idx] is not None
                    episodes = [int(x) for x in context_window_episodes]
                    context_window_text = MULTIDOC_PROMPT + "\n\n".join(
                        [
                            f"[START OF EPISODE]\n{transcripts[episode]}\n[END OF EPISODE]"
                            for episode in episodes
                        ]
                    )
                    output_dict = {
                        "id": generate_unique_uuid(
                            campaign + questions[idx] + "_".join(map(str, episodes))
                        ),
                        "context_window_id": generate_unique_uuid(
                            campaign + "_".join(map(str, episodes))
                        ),
                        "context_window_text": context_window_text,
                        "question": questions[idx],
                        "answer": answers[idx],
                        "question_type": types[idx],
                        "episodes": episodes,
                        "campaign": campaign,
                    }
                    f.write(json.dumps(output_dict) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
