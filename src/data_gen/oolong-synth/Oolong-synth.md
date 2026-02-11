# Creating Oolong-real
We use data from 10 ICL datasets, which we have validated by confirming that strong models can solve each question in a zeroshot setting. (See the paper for further details). The validated subset of each dataset is available in `validated_data/`

## Generating data
You can generate Oolong-synth questions by calling the script `generate_dataset.py` with the dataset and context lengths (2^x) of your choice, e.g.
```
cd src/data_gen/oolong-synth
python generate_dataset.py --dataset negation --context_windows_at_each_len 10 --context_lens 14 15 16 17 
```

will generate 10 context windows and associated questions for the negation dataset, for context lengths 16K (2^14) to 131K (2^17).

## Rebalancing questions
For the final Oolong-synth, we subsample the generated questions down to 50 for each context window and roughly match the composition of question types across datasets. To perform this post-processing step, once you have generated data for each dataset of interest, you can run `subsample.py`.
