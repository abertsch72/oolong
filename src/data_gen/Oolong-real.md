# Compiling Oolong-real

We use the episode transcripts released by [Rameshkumar and Bailey](https://aclanthology.org/2020.acl-main.459/). For episode statistics, we use the data compiled by [CritRoleStats](https://www.critrolestats.com/).

## Preparing transcripts

Download the JSON files from the [original repo](https://github.com/RevanthRameshkumar/CRD3/). We convert the transcripts into plain text format with one utterance per line (labeled with player name).

```bash
RAW="CRD3/data/cleaned\ data"
OUTPUT="data_gen/oolong-real/episodes"
python src/data_gen/prepare_cr_transcripts.py \
    --data-dir $RAW \
    --output-dir $OUTPUT
```

## Preparing roll and spell stats

Download the CSV files from CritRoleStats for rolls and spells.

### Rolls

Download csv files for [campaign 1](https://docs.google.com/spreadsheets/d/1OEg29XbL_YpO0m5JrLQpOPYTnxVsIg8iP67EYUrtRJg/edit?gid=1355947030#gid=1355947030) and [campaign 2](https://docs.google.com/spreadsheets/d/1FFuw5c6Hk1NUlHv2Wvr5b9AElLA51KtRl9ZruPU8r9k/edit?gid=770886437#gid=770886437).

Add column headers to each file, `["Episode", "Time", "Character", "Type", "of", "Roll", "Total", "Value", "Natural", "Value", "Crit?", "Damage", "Dealt", "#", "Kills", "Notes"]`.

Save these files to `data_gen/oolong-real/stats/campaign1/rolls.tsv` and `data_gen/oolong-real/stats/campaign2/rolls.tsv`.

### Spells

Download spells cast in each episode from these spreadsheets [campaign1](https://docs.google.com/spreadsheets/d/1Y7FB0rEUX8Ik0MfGUtsdItoFWcvlgpGVcSJ0l9-dGDw/edit?gid=0#gid=0) and [campaign2](https://docs.google.com/spreadsheets/d/1KXdTxewHmnBr6XTsX5pbUqZ9ObR4inZsgbaCQbMl7lc/edit?gid=1546235505#gid=1546235505). Merge stats in each campaign into a single CSV file. Example script for merging spell cast in campaign 2,

```bash
# episodes 20-46 have an extra column for old timestamps, remove them
for r in $(seq 20 46); do
    awk -F $'\t' -v OFS=$'\t' '
    {
    $1 = ""
    sub(/^\t/, "")
    print
    }
    ' Spells\ Cast\ -\ Wildemount\ -\ 2-$r.tsv > tmp.tsv && mv tmp.tsv Spells\ Cast\ -\ Wildemount\ -\ 2-$r.tsv
done
```

```bash
{
    read -r hdr < <(head -n1 "$(ls Spells\ Cast\ -\ Wildemount\ -\ *.tsv | head -1)")
    printf "episode\t%s\n" "$hdr"
    for f in Spells\ Cast\ -\ Wildemount\ -\ *.tsv; do
        ep=${f##* - 2-}
        ep=${ep%.tsv}
        tail -n +2 "$f" | awk -v ep="$ep" 'BEGIN{OFS="\t"}{print ep,$0}'
    done
} > spells.tsv
```

Add this column header to `spells.tsv`: `["Episode", "Time", "Character", "Spell", "Base", "Lvl", "Cast", "At", "Notes"]`.

Save these files to `data_gen/oolong-real/stats/campaign1/spells.tsv` and `data_gen/oolong-real/stats/campaign2/spells.tsv`.

## Generate question-answer pairs

We use campaign 1 and campaign 2 as test and validation splits respectively. To generate the toy dataset, uncomment the `--toy-dataset` flag in the commands below. Toy dataset only includes a small number of examples per question type.

```bash
DIR=data_gen/oolong-real
STATS_DIR=$DIR/stats
TRANSCRIPTS_DIR=$DIR/episodes
OUTPUT_DIR=$DIR/qa_pairs

CAMPAIGN=campaign1;
SPLIT=test;
python src/data_gen/prepare_cr_questions.py \
    --stats-dir $STATS_DIR \
    --transcripts-dir $TRANSCRIPTS_DIR \
    --output-dir $OUTPUT_DIR \
    --campaign $CAMPAIGN \
    --split $SPLIT \
    # --toy-dataset

CAMPAIGN=campaign2;
SPLIT=validation;
python src/data_gen/prepare_cr_questions.py \
    --stats-dir $STATS_DIR \
    --transcripts-dir $TRANSCRIPTS_DIR \
    --output-dir $OUTPUT_DIR \
    --campaign $CAMPAIGN \
    --split $SPLIT \
    # --toy-dataset
```
