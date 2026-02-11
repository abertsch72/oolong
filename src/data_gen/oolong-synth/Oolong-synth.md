# Creating Oolong-real
We use data from 10 ICL datasets, which we have validated by confirming that strong models can solve each question in a zeroshot setting. (See the paper for further details). The validated subset of each dataset is available in `validated_data/`

## Generating data
You can generate Oolong-synth questions by calling the script `generate_dataset.py` with the dataset and context lengths of your choice, e.g.
```
cd src/data_gen/oolong-synth
python generate_dataset.py --dataset negation --context_windows_at_each_len 10 --context_lens 14 15 16 17 
```
