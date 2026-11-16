# GoEmotions linear baselines

Code for the paper "How Far Can Linear Models Go in Fine-Grained Emotion Classification? A Reproducible Study on GoEmotions".

## Files
- `goemotions_baselines.py`: trains all TF-IDF models and writes `results.json`
- `sanity_checks.py`: shuffled-label control, manual macro-F1 recomputation, split-overlap check
- `results.json`: the results file behind every number in the paper

## Setup
Run in an empty folder, then place the scripts beside the downloaded files.

```
B=https://raw.githubusercontent.com/google-research/google-research/master/goemotions
for f in train.tsv dev.tsv test.tsv emotions.txt ekman_mapping.json sentiment_mapping.json; do curl -sSO $B/data/$f; done
pip install numpy scipy pandas scikit-learn matplotlib
```

The paper used scikit-learn 1.8.0.

## Run
```
python goemotions_baselines.py
```
Takes about seven minutes on one CPU core.

For the sanity controls:
```
cp goemotions_baselines.py run.py
python sanity_checks.py
```

## Data
The GoEmotions data are not included here. They are downloaded from the original Google Research repository by the commands above.
