# Multivariate benchmark results

17 equal-length datasets of the UEA multivariate archive, the archive's
train/test splits, test accuracy. Settings in `README.md`; nothing is
tuned per dataset. Bold marks the best accuracy on a dataset.

| Dataset | 1NN-ED | 1NN-DTW_I | 1NN-DTW_D | RandomForest | SoftDTWRSLVQ | SoftDTWMRSLVQ | SoftDTWLMRSLVQ |
|---|---|---|---|---|---|---|---|
| ArticularyWordRecognition | 0.970 | 0.977 | **0.987** | 0.977 | 0.983 | 0.973 | 0.980 |
| AtrialFibrillation | 0.267 | **0.333** | 0.200 | 0.200 | 0.200 | 0.200 | 0.200 |
| BasicMotions | 0.600 | **1.000** | 0.975 | 0.925 | **1.000** | 0.975 | 0.950 |
| Epilepsy | 0.630 | **0.964** | **0.964** | 0.797 | 0.935 | 0.957 | 0.942 |
| ERing | 0.944 | 0.878 | 0.915 | **0.959** | 0.915 | 0.889 | 0.915 |
| FingerMovements | 0.550 | 0.480 | 0.530 | **0.600** | 0.490 | 0.590 | 0.560 |
| HandMovementDirection | 0.257 | 0.311 | 0.189 | **0.514** | 0.473 | 0.500 | 0.446 |
| Handwriting | 0.332 | 0.505 | **0.607** | 0.221 | 0.574 | 0.595 | 0.565 |
| Libras | 0.828 | **0.894** | 0.872 | 0.733 | 0.850 | 0.783 | 0.817 |
| LSST | 0.451 | **0.572** | 0.551 | 0.536 | 0.386 | 0.392 | 0.388 |
| NATOPS | 0.839 | 0.861 | 0.883 | 0.844 | **0.928** | 0.922 | 0.922 |
| PenDigits | **0.977** | 0.941 | **0.977** | 0.966 | 0.953 | 0.958 | 0.963 |
| RacketSports | 0.822 | 0.822 | 0.803 | 0.895 | 0.855 | **0.914** | 0.849 |
| SelfRegulationSCP1 | 0.778 | 0.765 | 0.775 | 0.819 | **0.863** | **0.863** | 0.809 |
| SelfRegulationSCP2 | 0.483 | 0.533 | **0.539** | 0.461 | 0.506 | 0.517 | 0.489 |
| StandWalkJump | 0.200 | **0.400** | 0.200 | 0.333 | 0.267 | 0.133 | 0.333 |
| UWaveGestureLibrary | 0.875 | 0.856 | 0.903 | 0.866 | 0.887 | 0.887 | **0.912** |
| Mean | 0.636 | 0.711 | 0.698 | 0.685 | 0.710 | 0.709 | 0.708 |
| Better / equal / worse than 1NN-DTW_D | 7 / 1 / 9 | 7 / 1 / 9 | 0 / 17 / 0 | 6 / 1 / 10 | 6 / 2 / 9 | 5 / 2 / 10 | 7 / 2 / 8 |

## Statistics

| Method | Mean accuracy | Mean rank (1 = best) | Against `SoftDTWRSLVQ`: better / equal / worse | Wilcoxon p (Holm) |
|---|---|---|---|---|
| `SoftDTWRSLVQ` | 0.710 | 3.68 | – | – |
| `SoftDTWMRSLVQ` | 0.709 | 3.76 | 8 / 3 / 6 | 1.00 |
| `SoftDTWLMRSLVQ` | 0.708 | 3.88 | 6 / 2 / 9 | 1.00 |
| 1NN-DTW_I | 0.711 | 4.06 | 6 / 1 / 10 | 1.00 |
| 1NN-DTW_D | 0.698 | 3.50 | 9 / 2 / 6 | 1.00 |
| RandomForest | 0.685 | 4.18 | 7 / 1 / 9 | 1.00 |
| 1NN-ED | 0.636 | 4.94 | 5 / 0 / 12 | 0.43 |

The Friedman test finds no difference between the seven methods
(p = 0.53). The learned channel metric neither helps nor hurts accuracy
on average: `SoftDTWMRSLVQ` gains most on FingerMovements (+0.100),
RacketSports (+0.059) and HandMovementDirection (+0.027), and loses most
on StandWalkJump (−0.133, two of its 15 test series) and Libras (−0.067). Its value on these
data is the channel relevances it reports, not accuracy.
