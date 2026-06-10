# Paired-ensemble probing statistics (init contrast, timestep-shared attention-pooled probe)

- N = 100 ensemble members
- M_train = 1024 sequences
- Control contrast: init vs true (init = seed-deterministic untrained model)
- Probe: timestep-shared attention-pooled
- Bonferroni reference (per contrast): k=27, alpha_cell = 1.85e-03

## Per-cell paired statistics

| target | layer | mean Δ | 95% interval | W | p (sign) | Bonf? |
|---|---:|---:|---|---:|---:|:---:|
| C_star | 0 | +0.0090 | [-0.0130, +0.0313] | 86/100 | 4.14e-14 | ✓ |
| C_star | 1 | +0.7756 | [+0.5618, +0.9747] | 100/100 | 7.89e-31 | ✓ |
| C_star | 2 | +2.0764 | [+1.6262, +2.3324] | 100/100 | 7.89e-31 | ✓ |
| C_star | 3 | +2.6404 | [+2.4308, +2.8223] | 100/100 | 7.89e-31 | ✓ |
| C_star | 4 | +2.8167 | [+2.6926, +2.9340] | 100/100 | 7.89e-31 | ✓ |
| C_star | 5 | +2.8694 | [+2.7247, +2.9719] | 100/100 | 7.89e-31 | ✓ |
| C_star | 6 | +2.8528 | [+2.7064, +2.9752] | 100/100 | 7.89e-31 | ✓ |
| C_star | 7 | +2.8051 | [+2.6518, +2.9300] | 100/100 | 7.89e-31 | ✓ |
| C_star | 8 | +2.3364 | [+2.0533, +2.6330] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 0 | +13383.9516 | [+5034.8227, +23512.3589] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 1 | +34862.0804 | [+24144.5892, +45182.3450] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 2 | +39587.4101 | [+25465.1233, +52829.2414] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 3 | +44044.2747 | [+28681.8392, +58718.7027] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 4 | +45243.6256 | [+31775.7509, +59881.7907] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 5 | +44792.3545 | [+32456.3578, +57886.8320] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 6 | +43876.9225 | [+32557.3178, +57347.1579] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 7 | +43124.0183 | [+32996.6278, +58143.9343] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 8 | +41481.4216 | [+31334.7813, +54163.2816] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 0 | +0.0001 | [-0.0032, +0.0046] | 51/100 | 4.60e-01 |  |
| sigma_mle_sqrt | 1 | +0.1519 | [+0.1104, +0.1906] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 2 | +0.3947 | [+0.2877, +0.4424] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 3 | +0.5095 | [+0.4717, +0.5395] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 4 | +0.5422 | [+0.5190, +0.5593] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 5 | +0.5501 | [+0.5335, +0.5656] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 6 | +0.5500 | [+0.5294, +0.5655] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 7 | +0.5423 | [+0.5125, +0.5599] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 8 | +0.4784 | [+0.4376, +0.5189] | 100/100 | 7.89e-31 | ✓ |

Δ = test_loss(init) − test_loss(true) per member; positive Δ means the trained representation produces lower test error.
