# Time-series benchmark

Compares `SoftDTWRSLVQ` with standard time-series baselines on 24
datasets of the UCR time series archive (2018 release).

## Protocol

| Step | Choice |
|---|---|
| Datasets | 24 univariate datasets of the UCR archive (`DATASETS` in `run.py`), downloaded from timeseriesclassification.com |
| Split | The archive's own train/test split; every series is z-normalised |
| Baselines | 1-nearest neighbour under the Euclidean distance, under DTW, and under DTW with a Sakoe–Chiba band of 10 %; an RBF SVM and a random forest (500 trees) at their scikit-learn defaults; `WRSLVQ` (2 prototypes per class) |
| `SoftDTWRSLVQ` | `max_iter=150`, `lr=0.05`, the same on every dataset; three settings: `gamma=0.1` with 1 prototype per class (the defaults), `gamma=1` with 1, and `gamma=0.1` with 2. Mini-batches of 64 when a training set holds more than 20 000 values |
| Metric | Test accuracy |

Nothing is tuned. The 1-nearest-neighbour baselines reproduce the error
rates published with the archive, which checks the data and the DTW
implementation.

## Running it

```bash
# Data are downloaded once into ~/.cache/prosemble-ucr
# (override with PROSEMBLE_UCR_DATA).
python benchmarks/time_series/run.py --out results.jsonl

# In parallel, one process per shard:
for i in 0 1 2 3; do
  python benchmarks/time_series/run.py --out results_$i.jsonl --shard $i --shards 4 &
done; wait; cat results_*.jsonl > results.jsonl

python benchmarks/time_series/run.py --summary results.jsonl
```

An interrupted run resumes: results already in the output file are
skipped. Soft-DTW costs time proportional to the product of the series
lengths, so the datasets with the longest series (Lightning2, Herring,
Meat, OliveOil) take the longest.

Memory grows with that product too. Most of the DTW jobs need less than
1.5 GB, but one process can reach 6 GB on Lightning2 (1NN-DTW, and
`SoftDTWRSLVQ` with 2 prototypes per class), and 4–5 GB on Lightning7,
Herring, OliveOil and Meat. Choose the number of parallel processes by
the memory available, not by the number of cores.

`RESULTS.md` is the report of a full run.

## Multivariate benchmark

`run_multivariate.py` compares the soft-DTW models on 17 equal-length
datasets of the UEA multivariate archive (2018 release).

| Step | Choice |
|---|---|
| Datasets | `DATASETS` in `run_multivariate.py`: the equal-length UEA datasets with fewer than 61 channels, except Cricket (one soft fit takes about twelve hours on a CPU thread) |
| Split | The archive's own train/test split; series as distributed |
| Baselines | 1-nearest neighbour under the Euclidean distance, under independent DTW (summed over channels) and dependent DTW (multichannel ground cost); a random forest (500 trees) |
| Soft-DTW models | `SoftDTWRSLVQ`, `SoftDTWMRSLVQ`, `SoftDTWLMRSLVQ`: `gamma` equal to the number of channels, one prototype per class, `max_iter=150`, `lr=0.05`, each channel standardised with training-set statistics; the same on every dataset |
| Metric | Test accuracy |

```bash
# Data are downloaded once into ~/.cache/prosemble-uea
# (override with PROSEMBLE_UEA_DATA).
python benchmarks/time_series/run_multivariate.py --out multivariate.jsonl
python benchmarks/time_series/run_multivariate.py --summary multivariate.jsonl
```

`--datasets`, `--methods`, `--shard` and `--shards` split the run into
jobs. The long series need the most memory and time: StandWalkJump
(length 2500), SelfRegulationSCP2 (1152) and SelfRegulationSCP1 (896).
`RESULTS_multivariate.md` is the report of a full run.
