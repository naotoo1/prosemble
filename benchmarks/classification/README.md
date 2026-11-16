# Classification benchmark

A fixed, repeatable protocol for comparing prosemble's classifiers with
standard baselines over many datasets, with the statistical tests of
Demšar (2006).

## Protocol

| Step | Choice |
|---|---|
| Datasets | 36 OpenML datasets (`suite.py`), fixed by id: numeric features only, no missing values, at least 10 samples per class |
| Resampling | 2 repeats of stratified 5-fold cross-validation (10 train/test splits per dataset) |
| Preprocessing | Standardisation fitted on the training part of each split only |
| Baselines | Logistic regression, k-nearest neighbours and an RBF SVM, each tuned by a grid search over an inner, shuffled, stratified 3-fold split of the training part; random forest (500 trees) and histogram gradient boosting at their defaults |
| prosemble models | Fixed settings, identical on every dataset (no per-dataset tuning): 200 iterations, learning rate 0.01, 2 prototypes per class (1 for `DKLMRSLVQ`) |
| Probabilities | Soft prosemble models are calibrated with `calibrate(X_train, y_train, cv=3)`; the SVM uses Platt scaling |
| Metrics per split | Accuracy, balanced accuracy, log-loss, expected calibration error, accuracy on the 80 % most confident test samples, fit time |
| Statistics | Per metric, the mean over the 10 splits of a dataset is one observation. Friedman test and Iman–Davenport F test of equal mean ranks; Nemenyi critical difference; two-sided Wilcoxon signed-rank tests of one reference method against every other, Holm-adjusted |

The baselines are tuned and the prosemble models are not, so the
comparison is conservative for prosemble.

## Running it

```bash
# Datasets are downloaded once into ~/.cache/prosemble-benchmark
# (override with PROSEMBLE_BENCHMARK_DATA).
python benchmarks/classification/run.py --out results.jsonl

# In parallel: one process per shard, all appending to their own file.
for i in 0 1 2 3; do
  python benchmarks/classification/run.py --out results_$i.jsonl --shard $i --shards 4 &
done; wait; cat results_*.jsonl > results.jsonl

python benchmarks/classification/report.py results.jsonl --reference DKLMRSLVQ > RESULTS.md
```

`RESULTS.md` is the report of a full run.

An interrupted run resumes: splits already in the output file are
skipped. A split that raises an error is recorded and listed at the end
of the report; datasets with a missing method are left out of that
metric's statistics.

## Reading the report

- **Mean rank** is the average rank over datasets (1 = best). Two
  methods whose mean ranks differ by less than the Nemenyi critical
  difference are not significantly different.
- **Wins / ties / losses** count datasets on which the reference method
  is better, equal or worse; the Wilcoxon p-value tests whether the
  differences are centred on zero.
- A Friedman p-value above 0.05 means the data do not show any
  difference between the methods; pairwise results should then not be
  read as significant.

## References

- Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple
  Data Sets. *Journal of Machine Learning Research*, 7, 1–30.
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test
  Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
