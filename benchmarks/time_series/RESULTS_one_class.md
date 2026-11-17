# One-class benchmark results

Each class of a dataset in turn is the normal class; a detector is fitted to the
training series of that class only and scores every test series. AUROC for telling
the normal test series from the others, averaged over the classes of a dataset.
Settings in `README.md`; nothing is tuned per dataset. Bold marks the best AUROC
on a dataset.

The datasets that informed the design of `OCAlignmentDensity` (the choice of
per-position variances with empirical-Bayes shrinkage and of three prototypes) are
reported separately: CBF, ECG200, GunPoint, ItalyPowerDemand, SyntheticControl,
TwoLeadECG, Trace, Meat, OliveOil and Herring (UCR); ERing, BasicMotions,
RacketSports and NATOPS (UEA).

## UCR archive, univariate

| Dataset | 1NN-ED | 1NN-DTW | OC-SVM | IsolationForest | OCAlignmentDensity |
|---|---|---|---|---|---|
| ArrowHead | **0.768** | 0.741 | 0.691 | 0.752 | 0.751 |
| BeetleFly | 0.705 | 0.710 | 0.755 | 0.735 | **0.850** |
| BirdChicken | 0.575 | 0.690 | 0.705 | 0.590 | **0.765** |
| CBF | 0.937 | 0.991 | 0.911 | 0.845 | **0.992** |
| Coffee | 0.962 | **0.974** | 0.903 | 0.818 | 0.915 |
| DistalPhalanxOutlineCorrect | 0.579 | 0.577 | 0.548 | 0.610 | **0.641** |
| ECG200 | **0.822** | 0.683 | 0.720 | 0.713 | 0.687 |
| ECGFiveDays | **0.826** | 0.790 | 0.777 | 0.694 | 0.750 |
| FaceFour | 0.892 | 0.812 | **0.899** | 0.824 | 0.846 |
| GunPoint | 0.846 | 0.843 | 0.536 | 0.729 | **0.904** |
| Herring | 0.520 | 0.527 | 0.496 | 0.516 | **0.537** |
| ItalyPowerDemand | 0.887 | 0.802 | **0.939** | 0.802 | 0.871 |
| Lightning2 | 0.633 | 0.706 | 0.580 | 0.518 | **0.734** |
| Lightning7 | 0.742 | 0.846 | 0.728 | 0.848 | **0.865** |
| Meat | **0.982** | 0.975 | 0.929 | 0.922 | 0.966 |
| MoteStrain | 0.754 | 0.704 | 0.663 | 0.799 | **0.810** |
| OliveOil | 0.900 | 0.904 | 0.780 | 0.707 | **0.906** |
| Plane | 0.996 | **1.000** | 0.981 | 0.962 | **1.000** |
| SonyAIBORobotSurface1 | 0.751 | 0.703 | 0.737 | 0.632 | **0.779** |
| SyntheticControl | 0.888 | **0.985** | 0.859 | 0.863 | 0.963 |
| ToeSegmentation1 | 0.618 | 0.614 | 0.596 | 0.517 | **0.624** |
| Trace | 0.936 | **0.999** | 0.817 | 0.911 | 0.990 |
| TwoLeadECG | 0.780 | **0.859** | 0.575 | 0.632 | 0.833 |
| Wine | 0.570 | 0.569 | 0.482 | 0.580 | **0.680** |
| Mean | 0.786 | 0.792 | 0.734 | 0.730 | 0.819 |
| Better / equal / worse than 1NN-DTW | 12 / 0 / 12 | 0 / 24 / 0 | 6 / 0 / 18 | 8 / 1 / 15 | 17 / 1 / 6 |

### Datasets not used for design choices

14 datasets; Friedman test over the 5 methods: p = 0.0051.

| Method | Mean AUROC | Mean rank (1 = best) | OCAlignmentDensity better / equal / worse | Wilcoxon p (Holm) |
|---|---|---|---|---|
| **OCAlignmentDensity** | 0.787 | 1.71 | – | – |
| 1NN-DTW | 0.745 | 3.14 | 11 / 0 / 3 | 0.041 |
| 1NN-ED | 0.741 | 2.79 | 10 / 0 / 4 | 0.068 |
| OC-SVM | 0.718 | 3.71 | 12 / 0 / 2 | 0.012 |
| IsolationForest | 0.706 | 3.64 | 13 / 0 / 1 | 0.00098 |

### All datasets

24 datasets; Friedman test over the 5 methods: p = 9e-06.

| Method | Mean AUROC | Mean rank (1 = best) | OCAlignmentDensity better / equal / worse | Wilcoxon p (Holm) |
|---|---|---|---|---|
| **OCAlignmentDensity** | 0.819 | 1.83 | – | – |
| 1NN-DTW | 0.792 | 2.83 | 17 / 0 / 7 | 0.03 |
| 1NN-ED | 0.786 | 2.62 | 17 / 0 / 7 | 0.03 |
| OC-SVM | 0.734 | 3.83 | 20 / 0 / 4 | 0.00062 |
| IsolationForest | 0.730 | 3.88 | 22 / 0 / 2 | 9.1e-06 |

## UEA archive, multivariate

| Dataset | 1NN-ED | 1NN-DTW | OC-SVM | IsolationForest | OCAlignmentDensity |
|---|---|---|---|---|---|
| ArticularyWordRecognition | 0.993 | 0.996 | 0.993 | 0.980 | **0.998** |
| AtrialFibrillation | 0.493 | 0.473 | **0.500** | 0.497 | 0.427 |
| BasicMotions | 0.666 | 0.806 | 0.586 | 0.596 | **0.814** |
| Epilepsy | 0.772 | **0.911** | 0.741 | 0.692 | 0.870 |
| ERing | 0.976 | **0.984** | 0.978 | 0.952 | 0.973 |
| FingerMovements | **0.535** | 0.521 | 0.478 | 0.505 | 0.505 |
| HandMovementDirection | 0.522 | 0.482 | 0.530 | 0.529 | **0.531** |
| Handwriting | 0.683 | **0.892** | 0.659 | 0.561 | 0.844 |
| Libras | 0.970 | **0.988** | 0.934 | 0.807 | 0.968 |
| LSST | 0.455 | 0.463 | **0.492** | 0.424 | 0.447 |
| NATOPS | 0.905 | 0.920 | 0.859 | 0.847 | **0.934** |
| PenDigits | 0.995 | **0.996** | 0.979 | 0.978 | 0.986 |
| RacketSports | 0.807 | 0.907 | 0.692 | 0.677 | **0.915** |
| SelfRegulationSCP1 | 0.660 | 0.650 | **0.693** | 0.615 | 0.675 |
| SelfRegulationSCP2 | 0.497 | **0.528** | 0.487 | 0.495 | 0.504 |
| StandWalkJump | 0.507 | 0.433 | **0.527** | 0.370 | 0.480 |
| UWaveGestureLibrary | 0.966 | **0.976** | 0.974 | 0.952 | 0.974 |
| Mean | 0.730 | 0.760 | 0.712 | 0.675 | 0.756 |
| Better / equal / worse than 1NN-DTW | 5 / 0 / 12 | 0 / 17 / 0 | 5 / 0 / 12 | 2 / 0 / 15 | 7 / 0 / 10 |

### Datasets not used for design choices

13 datasets; Friedman test over the 5 methods: p = 0.0058.

| Method | Mean AUROC | Mean rank (1 = best) | OCAlignmentDensity better / equal / worse | Wilcoxon p (Holm) |
|---|---|---|---|---|
| 1NN-DTW | 0.716 | 2.23 | 4 / 0 / 9 | 1 |
| **OCAlignmentDensity** | 0.708 | 2.65 | – | – |
| 1NN-ED | 0.696 | 2.85 | 7 / 0 / 6 | 1 |
| OC-SVM | 0.691 | 2.85 | 8 / 0 / 5 | 1 |
| IsolationForest | 0.646 | 4.42 | 11 / 1 / 1 | 0.049 |

### All datasets

17 datasets; Friedman test over the 5 methods: p = 0.00013.

| Method | Mean AUROC | Mean rank (1 = best) | OCAlignmentDensity better / equal / worse | Wilcoxon p (Holm) |
|---|---|---|---|---|
| 1NN-DTW | 0.760 | 2.12 | 7 / 0 / 10 | 0.53 |
| **OCAlignmentDensity** | 0.756 | 2.44 | – | – |
| 1NN-ED | 0.730 | 2.88 | 10 / 0 / 7 | 0.53 |
| OC-SVM | 0.712 | 3.06 | 11 / 0 / 6 | 0.4 |
| IsolationForest | 0.675 | 4.50 | 15 / 1 / 1 | 0.0091 |
## Notes

- On the UCR archive `OCAlignmentDensity` has the highest mean AUROC and the best
  mean rank, and beats the nearest-neighbour detector under DTW on 11 of the 14
  datasets not used for design (Holm-adjusted Wilcoxon p = 0.041).
- On the UEA archive it is second to 1NN-DTW (0.708 against 0.716 on the 13 datasets
  not used for design; the difference is not significant) and ahead of the other
  detectors. On LSST, SelfRegulationSCP2, StandWalkJump and AtrialFibrillation every
  detector is near or below 0.5.
- The prototypes start at soft-DTW medoids, and EM finds a local optimum from there.
  With other initial medoids (a different random generator), the AUROC of a class
  with few training series moved by up to 0.2 (SonyAIBORobotSurface1 has 6 training
  series in one class, FaceFour 5 to 8 per class), and the dataset means by 0.01 at
  the median and by up to 0.13 (BeetleFly, 10 per class). Small training sets make the
  result depend on the start.
