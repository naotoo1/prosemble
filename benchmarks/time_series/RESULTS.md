# Time-series benchmark

Source: a full run of `run.py` (216 results: 24 datasets, 9 methods). Wins, ties and losses count datasets on which the `SoftDTWRSLVQ` setting is better, equal or worse; p-values are two-sided Wilcoxon signed-rank tests, Holm-adjusted over the comparisons in each table.

All three `SoftDTWRSLVQ` settings were fixed before the run. Each is compared with the six baselines on its own, so that no table compares a method with its own variants. Reporting only the best of the three on the same test sets would favour it; the tables give all three.

## `SoftDTWRSLVQ` against the baselines

24 datasets. Friedman chi-squared = 26.0, p = 0.00023; Iman-Davenport F = 5.06, p = 0.0001. Nemenyi critical difference of mean ranks (alpha = 0.05): 1.84.

| Method | Mean | Mean rank | SoftDTWRSLVQ: wins / ties / losses | Wilcoxon p (Holm) |
|---|---|---|---|---|
| **SoftDTWRSLVQ** | 0.862 | 3.12 | — | — |
| RandomForest | 0.827 | 3.33 | 14 / 1 / 9 | 0.436 |
| 1NN-DTW (band 0.1) | 0.831 | 3.46 | 12 / 4 / 8 | 0.436 |
| 1NN-DTW | 0.824 | 3.83 | 11 / 5 / 8 | 0.436 |
| WRSLVQ | 0.815 | 3.96 | 14 / 3 / 7 | 0.436 |
| 1NN-ED | 0.786 | 4.48 | 15 / 1 / 8 | 0.103 |
| SVM | 0.709 | 5.81 | 20 / 0 / 4 | 0.00193 |

## `SoftDTWRSLVQ (gamma 1)` against the baselines

24 datasets. Friedman chi-squared = 31.1, p = 2.4e-05; Iman-Davenport F = 6.34, p = 6.5e-06. Nemenyi critical difference of mean ranks (alpha = 0.05): 1.84.

| Method | Mean | Mean rank | SoftDTWRSLVQ (gamma 1): wins / ties / losses | Wilcoxon p (Holm) |
|---|---|---|---|---|
| **SoftDTWRSLVQ (gamma 1)** | 0.880 | 2.60 | — | — |
| RandomForest | 0.827 | 3.46 | 17 / 1 / 6 | 0.0397 |
| 1NN-DTW (band 0.1) | 0.831 | 3.52 | 14 / 3 / 7 | 0.0397 |
| 1NN-DTW | 0.824 | 4.00 | 16 / 3 / 5 | 0.0387 |
| WRSLVQ | 0.815 | 4.06 | 17 / 2 / 5 | 0.0387 |
| 1NN-ED | 0.786 | 4.54 | 16 / 2 / 6 | 0.0233 |
| SVM | 0.709 | 5.81 | 20 / 0 / 4 | 0.000383 |

## `SoftDTWRSLVQ (2 per class)` against the baselines

24 datasets. Friedman chi-squared = 27.2, p = 0.00013; Iman-Davenport F = 5.36, p = 5.3e-05. Nemenyi critical difference of mean ranks (alpha = 0.05): 1.84.

| Method | Mean | Mean rank | SoftDTWRSLVQ (2 per class): wins / ties / losses | Wilcoxon p (Holm) |
|---|---|---|---|---|
| **SoftDTWRSLVQ (2 per class)** | 0.846 | 2.94 | — | — |
| RandomForest | 0.827 | 3.42 | 16 / 1 / 7 | 0.176 |
| 1NN-DTW (band 0.1) | 0.831 | 3.50 | 13 / 4 / 7 | 0.213 |
| 1NN-DTW | 0.824 | 3.90 | 13 / 4 / 7 | 0.213 |
| WRSLVQ | 0.815 | 3.92 | 14 / 1 / 9 | 0.213 |
| 1NN-ED | 0.786 | 4.52 | 16 / 1 / 7 | 0.0672 |
| SVM | 0.709 | 5.81 | 19 / 2 / 3 | 0.00309 |

## All nine methods

24 datasets. Friedman chi-squared = 36.4, p = 1.5e-05; Iman-Davenport F = 5.39, p = 4.3e-06. Nemenyi critical difference of mean ranks (alpha = 0.05): 2.45.

| Method | Mean | Mean rank |
|---|---|---|
| SoftDTWRSLVQ (gamma 1) | 0.880 | 3.31 |
| SoftDTWRSLVQ (2 per class) | 0.846 | 4.15 |
| SoftDTWRSLVQ | 0.862 | 4.21 |
| 1NN-DTW (band 0.1) | 0.831 | 4.73 |
| RandomForest | 0.827 | 4.75 |
| 1NN-DTW | 0.824 | 5.19 |
| WRSLVQ | 0.815 | 5.31 |
| 1NN-ED | 0.786 | 5.88 |
| SVM | 0.709 | 7.48 |

## Accuracy per dataset

The best accuracy on each dataset is in bold.

| Dataset | 1NN-ED | 1NN-DTW | 1NN-DTW (band 0.1) | SVM | RandomForest | WRSLVQ | SoftDTWRSLVQ | SoftDTWRSLVQ (gamma 1) | SoftDTWRSLVQ (2 per class) |
|---|---|---|---|---|---|---|---|---|---|
| ArrowHead | **0.800** | 0.703 | 0.720 | 0.554 | 0.731 | **0.800** | 0.783 | 0.737 | 0.731 |
| BeetleFly | 0.750 | 0.700 | 0.700 | 0.850 | **0.900** | 0.850 | 0.700 | 0.750 | 0.700 |
| BirdChicken | 0.550 | 0.750 | 0.750 | 0.600 | 0.750 | 0.700 | 0.900 | **1.000** | 0.800 |
| CBF | 0.852 | **0.997** | 0.996 | 0.886 | 0.889 | 0.851 | 0.983 | 0.992 | 0.990 |
| Coffee | **1.000** | **1.000** | **1.000** | 0.929 | **1.000** | **1.000** | **1.000** | **1.000** | 0.964 |
| DistalPhalanxOutlineCorrect | 0.717 | 0.717 | 0.717 | 0.656 | 0.768 | 0.743 | 0.757 | **0.779** | 0.761 |
| ECG200 | 0.880 | 0.770 | 0.820 | 0.860 | 0.810 | **0.910** | 0.770 | 0.800 | 0.870 |
| ECGFiveDays | 0.797 | 0.768 | 0.787 | 0.612 | 0.812 | 0.804 | 0.913 | **0.934** | 0.882 |
| FaceFour | 0.784 | 0.830 | 0.830 | 0.500 | 0.795 | 0.818 | 0.705 | **0.886** | 0.818 |
| GunPoint | 0.913 | 0.907 | 0.940 | 0.767 | 0.940 | 0.947 | **0.993** | **0.993** | 0.987 |
| Herring | 0.516 | 0.531 | 0.531 | 0.594 | 0.609 | 0.641 | 0.656 | **0.703** | 0.594 |
| ItalyPowerDemand | 0.955 | 0.950 | 0.952 | 0.956 | **0.966** | 0.964 | 0.915 | 0.915 | 0.917 |
| Lightning2 | 0.754 | **0.869** | **0.869** | 0.705 | 0.754 | 0.656 | 0.770 | 0.770 | 0.803 |
| Lightning7 | 0.575 | 0.726 | 0.740 | 0.562 | 0.740 | 0.658 | **0.877** | 0.836 | 0.849 |
| Meat | **0.933** | **0.933** | **0.933** | 0.883 | 0.917 | 0.900 | 0.900 | 0.900 | **0.933** |
| MoteStrain | 0.879 | 0.835 | 0.868 | 0.868 | **0.886** | 0.862 | 0.826 | 0.866 | 0.855 |
| OliveOil | 0.867 | 0.833 | 0.833 | 0.400 | **0.900** | 0.800 | 0.767 | 0.767 | 0.400 |
| Plane | 0.962 | **1.000** | **1.000** | 0.962 | 0.990 | 0.971 | **1.000** | **1.000** | **1.000** |
| SonyAIBORobotSurface1 | 0.696 | 0.725 | 0.725 | 0.676 | 0.664 | 0.739 | **0.752** | 0.744 | 0.737 |
| SyntheticControl | 0.880 | 0.993 | 0.983 | 0.977 | 0.957 | 0.947 | 0.990 | **1.000** | 0.990 |
| ToeSegmentation1 | 0.680 | 0.772 | 0.776 | 0.632 | 0.662 | 0.605 | **0.921** | 0.908 | 0.855 |
| Trace | 0.760 | **1.000** | **1.000** | 0.520 | 0.830 | 0.760 | **1.000** | **1.000** | **1.000** |
| TwoLeadECG | 0.747 | 0.904 | 0.894 | 0.574 | 0.734 | 0.816 | **0.984** | 0.960 | 0.961 |
| Wine | 0.611 | 0.574 | 0.574 | 0.500 | 0.833 | 0.815 | 0.815 | 0.889 | **0.907** |
| Mean | 0.786 | 0.824 | 0.831 | 0.709 | 0.827 | 0.815 | 0.862 | 0.880 | 0.846 |
| Better / equal / worse than 1NN-DTW | 9 / 3 / 12 | 0 / 24 / 0 | 8 / 13 / 3 | 5 / 0 / 19 | 12 / 2 / 10 | 11 / 1 / 12 | 11 / 5 / 8 | 16 / 3 / 5 | 13 / 4 / 7 |

## Checks

The test accuracies of 1NN-ED and 1NN-DTW agree, on all 24 datasets, with the error rates published with the archive (`ED (w=0)` and `DTW (w=100)` in its `DataSummary.csv`).

## Running time

Total time to fit and predict over the 24 datasets (`fit_time` in the results). Up to 16 jobs ran at once on one laptop CPU, so the times show the relative cost only.

| Method | Minutes |
|---|---|
| 1NN-ED | 0.0 |
| 1NN-DTW | 8.6 |
| 1NN-DTW (band 0.1) | 8.6 |
| SVM | 0.0 |
| RandomForest | 0.5 |
| WRSLVQ | 4.9 |
| SoftDTWRSLVQ | 167.4 |
| SoftDTWRSLVQ (gamma 1) | 176.0 |
| SoftDTWRSLVQ (2 per class) | 346.2 |
