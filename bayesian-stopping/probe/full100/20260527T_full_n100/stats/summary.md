# Paired-ensemble probing statistics

- N = 100 ensemble members
- M_train = 1024 sequences
- Control contrast: perm_glob vs true
- Probe: timestep-shared simple linear, σ-normalized MSE
- Bonferroni reference: k=27, alpha_cell = 1.85e-03

## Per-cell paired statistics

| target | layer | mean Δ | 95% CI | W | p (sign) | Bonf? |
|---|---:|---:|---|---:|---:|:---:|
| C_star | 0 | +0.0841 | [+0.0346, +0.1126] | 99/100 | 7.97e-29 | ✓ |
| C_star | 1 | +0.4459 | [+0.1644, +0.6923] | 98/100 | 3.98e-27 | ✓ |
| C_star | 2 | +1.5853 | [+0.0805, +2.1145] | 98/100 | 3.98e-27 | ✓ |
| C_star | 3 | +2.3144 | [+0.2475, +2.7908] | 98/100 | 3.98e-27 | ✓ |
| C_star | 4 | +2.5701 | [+0.0504, +2.9870] | 97/100 | 1.32e-25 | ✓ |
| C_star | 5 | +2.5919 | [+0.2712, +2.9809] | 98/100 | 3.98e-27 | ✓ |
| C_star | 6 | +2.5368 | [+0.2606, +3.0394] | 98/100 | 3.98e-27 | ✓ |
| C_star | 7 | +2.4660 | [-0.1207, +2.9965] | 96/100 | 3.22e-24 | ✓ |
| C_star | 8 | +1.5803 | [-0.3666, +2.4367] | 96/100 | 3.22e-24 | ✓ |
| loglik_true_params_raw_mse | 0 | +2378.1575 | [+1821.8133, +2999.6359] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 1 | -16196.7103 | [-45192.5930, +3677.0031] | 6/100 | 1.00e+00 |  |
| loglik_true_params_raw_mse | 2 | -37360.2413 | [-68110.0520, -11765.5609] | 0/100 | 1.00e+00 |  |
| loglik_true_params_raw_mse | 3 | -30407.0981 | [-57489.6947, -5137.7441] | 1/100 | 1.00e+00 |  |
| loglik_true_params_raw_mse | 4 | -12752.7920 | [-42992.4000, +9360.4668] | 16/100 | 1.00e+00 |  |
| loglik_true_params_raw_mse | 5 | +1435.8705 | [-21500.3223, +16056.3027] | 60/100 | 2.84e-02 |  |
| loglik_true_params_raw_mse | 6 | +9701.4181 | [-2597.1049, +23571.5875] | 90/100 | 1.53e-17 | ✓ |
| loglik_true_params_raw_mse | 7 | +13799.2011 | [+4391.0119, +22493.3596] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 8 | +5515.0128 | [-1885.6221, +13538.4885] | 91/100 | 1.66e-18 | ✓ |
| sigma_mle_sqrt | 0 | +0.0024 | [-0.0001, +0.0065] | 97/100 | 1.32e-25 | ✓ |
| sigma_mle_sqrt | 1 | +0.0522 | [+0.0244, +0.0779] | 99/100 | 7.97e-29 | ✓ |
| sigma_mle_sqrt | 2 | +0.1883 | [+0.0908, +0.2805] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 3 | +0.3222 | [+0.2138, +0.3913] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 4 | +0.3912 | [+0.3212, +0.4500] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 5 | +0.4117 | [+0.3522, +0.4669] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 6 | +0.4042 | [+0.3381, +0.4641] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 7 | +0.3975 | [+0.3203, +0.4848] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 8 | +0.2726 | [+0.1625, +0.3779] | 99/100 | 7.97e-29 | ✓ |

Δ = M_perm_glob - M_true on test split (σ-normalized MSE).
Positive Δ means the true representation produces a lower test error.
