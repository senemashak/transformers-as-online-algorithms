# Paired-ensemble probing statistics (attention-pooled probe)

- N = 100 ensemble members
- M_train = 1024 sequences
- Control contrast: perm_glob vs true
- Probe: timestep-shared attention-pooled (linear FF), σ-normalized MSE
- Bonferroni reference: k=27, alpha_cell = 1.85e-03

## Per-cell paired statistics

| target | layer | mean Δ | 95% CI | W | p (sign) | Bonf? |
|---|---:|---:|---|---:|---:|:---:|
| C_star | 0 | +0.0079 | [-0.0097, +0.0250] | 86/100 | 4.14e-14 | ✓ |
| C_star | 1 | +0.7447 | [+0.5364, +0.9480] | 99/100 | 7.97e-29 | ✓ |
| C_star | 2 | +1.8715 | [+1.1196, +2.2546] | 100/100 | 7.89e-31 | ✓ |
| C_star | 3 | +2.1509 | [+1.2453, +2.6838] | 100/100 | 7.89e-31 | ✓ |
| C_star | 4 | +2.1084 | [+1.1356, +2.8155] | 100/100 | 7.89e-31 | ✓ |
| C_star | 5 | +1.9966 | [+1.1434, +2.8475] | 100/100 | 7.89e-31 | ✓ |
| C_star | 6 | +1.8519 | [+0.9905, +2.6042] | 100/100 | 7.89e-31 | ✓ |
| C_star | 7 | +1.7657 | [+1.0157, +2.5702] | 100/100 | 7.89e-31 | ✓ |
| C_star | 8 | +1.2942 | [+0.5485, +1.9983] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 0 | +13309.2803 | [+5092.9646, +23596.2321] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 1 | +56309.0854 | [+43598.6831, +69831.2567] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 2 | +71296.2293 | [+59015.6311, +81222.4610] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 3 | +75655.2676 | [+63365.9448, +85248.9321] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 4 | +76700.9977 | [+67000.9275, +84769.9519] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 5 | +77397.9787 | [+68823.5311, +85492.7091] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 6 | +77388.0815 | [+67814.9271, +84571.4151] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 7 | +77176.5546 | [+69282.6272, +83432.4869] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 8 | +76410.6634 | [+69096.6344, +82326.7701] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 0 | -0.0001 | [-0.0028, +0.0026] | 44/100 | 9.03e-01 |  |
| sigma_mle_sqrt | 1 | +0.1404 | [+0.0820, +0.1809] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 2 | +0.3349 | [+0.1578, +0.4117] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 3 | +0.3741 | [+0.1959, +0.5037] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 4 | +0.3463 | [+0.1620, +0.5020] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 5 | +0.3176 | [+0.1690, +0.5067] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 6 | +0.2952 | [+0.1685, +0.4590] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 7 | +0.2741 | [+0.1423, +0.4041] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 8 | +0.2215 | [+0.0776, +0.3730] | 100/100 | 7.89e-31 | ✓ |

Δ = M_perm_glob - M_true on test split (σ-normalized MSE), attention-pooled probe.
