# References

This page records the external datasets, papers and methods that provide context for AeroGuard-PHM.

## License

AeroGuard-PHM is released under Apache License 2.0. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Dataset

- NASA Prognostics Center of Excellence. C-MAPSS Turbofan Engine Degradation Simulation Data Set. Used as the simulated benchmark for remaining-useful-life evaluation.

## Sequence and Transformer Modelling

- Vaswani et al. "Attention Is All You Need." 2017. Transformer architecture reference.
- Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." 2023. Patch-based time-series Transformer inspiration.

## Predictive Maintenance and RUL

- Saxena and Goebel. Turbofan engine degradation simulation and prognostics benchmark documentation associated with NASA C-MAPSS.
- NASA prognostics scoring conventions for asymmetric late/early RUL prediction penalties.

## Physics-Guided and Safety-Aware Modelling

- Physics-guided machine-learning literature on monotonicity, smoothness, rate constraints and domain-informed regularization.
- Safety-aware model-selection practices separating point-prediction error from operational review and miss behavior.

## KAN Experiments

- Liu et al. "KAN: Kolmogorov-Arnold Networks." 2024. Method reference for the experimental AeroKAN residual-correction branch.

## Conformal Prediction

- Vovk, Gammerman and Shafer. "Algorithmic Learning in a Random World." 2005.
- Split conformal prediction literature for finite-sample uncertainty intervals calibrated on held-out residuals.

## Public-Release Notes

Raw C-MAPSS files are treated as local data and ignored by default. Cite the original dataset source when reproducing experiments.
