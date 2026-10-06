# UTWF simulation results

Synthetic data. 5 seeds per topology; 28 training days, 7 validation days (threshold selection), 14 test days. Mean ± std over seeds.

## Detection performance (test period)

| Topology | Method | F1 | Precision | Recall | False alarms / junction-day | Median delay (min) | ROC-AUC |
|---|---|---|---|---|---|---|---|
| grid | UTWF | 0.917 ± 0.045 | 0.964 | 0.881 | 0.015 | 15.0 | 0.982 |
| grid | UTWF (no spatial) | 0.929 ± 0.041 | 0.968 | 0.897 | 0.012 | 15.0 | 0.975 |
| grid | UTWF (no common-mode) | 0.731 ± 0.173 | 0.746 | 0.813 | 0.317 | 17.0 | 0.956 |
| grid | Historical average + CM | 0.922 ± 0.042 | 0.975 | 0.876 | 0.010 | 15.0 | 0.967 |
| grid | Historical average | 0.793 ± 0.096 | 0.807 | 0.821 | 0.125 | 17.0 | 0.943 |
| grid | Rolling z-score | 0.478 ± 0.035 | 0.428 | 0.552 | 0.199 | 10.0 | 0.658 |
| radial | UTWF | 0.921 ± 0.019 | 0.952 | 0.893 | 0.017 | 15.0 | 0.984 |
| radial | UTWF (no spatial) | 0.922 ± 0.018 | 0.975 | 0.877 | 0.007 | 15.0 | 0.974 |
| radial | UTWF (no common-mode) | 0.719 ± 0.176 | 0.824 | 0.726 | 0.231 | 19.0 | 0.944 |
| radial | Historical average + CM | 0.908 ± 0.022 | 0.942 | 0.883 | 0.019 | 15.0 | 0.967 |
| radial | Historical average | 0.720 ± 0.163 | 0.847 | 0.709 | 0.166 | 15.0 | 0.943 |
| radial | Rolling z-score | 0.439 ± 0.018 | 0.404 | 0.538 | 0.162 | 10.0 | 0.683 |
| organic | UTWF | 0.944 ± 0.022 | 0.962 | 0.928 | 0.022 | 15.0 | 0.982 |
| organic | UTWF (no spatial) | 0.911 ± 0.026 | 0.942 | 0.887 | 0.033 | 15.0 | 0.967 |
| organic | UTWF (no common-mode) | 0.745 ± 0.171 | 0.752 | 0.804 | 0.302 | 17.0 | 0.965 |
| organic | Historical average + CM | 0.905 ± 0.031 | 0.960 | 0.859 | 0.023 | 15.0 | 0.960 |
| organic | Historical average | 0.722 ± 0.167 | 0.744 | 0.774 | 0.331 | 15.0 | 0.952 |
| organic | Rolling z-score | 0.522 ± 0.032 | 0.557 | 0.506 | 0.195 | 10.0 | 0.653 |

## Junction model fit (clean test data)

| Topology | Median R² | Min R² |
|---|---|---|
| grid | 0.980 | 0.967 |
| radial | 0.974 | 0.956 |
| organic | 0.973 | 0.954 |

## City fingerprint

Leave-one-out nearest-centroid topology classification over 30 simulated cities: **80.0%**.

Note: topology differences in the fingerprint come from the simulator's assumptions (link lengths, network depth, demand placement), so this shows the fingerprint *can* separate structurally different networks, not that real cities differ this way.
