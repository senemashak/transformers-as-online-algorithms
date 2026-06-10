# Paired-ensemble probing statistics (init contrast, timestep-shared simple linear probe)

- N = 100 ensemble members
- M_train = 1024 sequences
- Control contrast: init vs true (init = seed-deterministic untrained model)
- Probe: timestep-shared simple linear
- Bonferroni reference (per contrast): k=27, alpha_cell = 1.85e-03

## Per-cell paired statistics

| target | layer | mean Δ | 95% interval | W | p (sign) | Bonf? |
|---|---:|---:|---|---:|---:|:---:|
| C_star | 0 | +0.2000 | [+0.0667, +2.8578] | 100/100 | 7.89e-31 | ✓ |
| C_star | 1 | +0.5402 | [+0.2874, +2.4839] | 100/100 | 7.89e-31 | ✓ |
| C_star | 2 | +1.7381 | [+1.2733, +2.3777] | 100/100 | 7.89e-31 | ✓ |
| C_star | 3 | +2.5421 | [+1.9867, +2.9602] | 100/100 | 7.89e-31 | ✓ |
| C_star | 4 | +2.8478 | [+2.3275, +3.1457] | 100/100 | 7.89e-31 | ✓ |
| C_star | 5 | +2.9069 | [+2.4271, +3.1884] | 100/100 | 7.89e-31 | ✓ |
| C_star | 6 | +2.8660 | [+2.2509, +3.2198] | 100/100 | 7.89e-31 | ✓ |
| C_star | 7 | +2.8415 | [+2.2424, +3.1677] | 100/100 | 7.89e-31 | ✓ |
| C_star | 8 | +1.9466 | [+0.8605, +2.6639] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 0 | +2595.4925 | [+1987.8766, +3243.4211] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 1 | +91917.2109 | [+70478.9660, +112246.2816] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 2 | +131602.9788 | [+104851.5109, +155727.6836] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 3 | +161948.0659 | [+133456.2629, +191001.0641] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 4 | +188662.5476 | [+158055.3777, +211109.6488] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 5 | +206303.8171 | [+182352.9947, +223207.9377] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 6 | +216170.6597 | [+202033.9789, +229950.4715] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 7 | +221193.7935 | [+210461.4143, +231312.8260] | 100/100 | 7.89e-31 | ✓ |
| loglik_true_params_raw_mse | 8 | +213447.0073 | [+203003.2775, +223217.3852] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 0 | +0.0005 | [-0.0280, +0.0047] | 93/100 | 1.36e-20 | ✓ |
| sigma_mle_sqrt | 1 | +0.0518 | [-0.0228, +0.0837] | 96/100 | 3.22e-24 | ✓ |
| sigma_mle_sqrt | 2 | +0.1968 | [+0.0411, +0.2892] | 98/100 | 3.98e-27 | ✓ |
| sigma_mle_sqrt | 3 | +0.3400 | [+0.1620, +0.4145] | 99/100 | 7.97e-29 | ✓ |
| sigma_mle_sqrt | 4 | +0.4223 | [+0.3133, +0.4760] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 5 | +0.4507 | [+0.3536, +0.5018] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 6 | +0.4553 | [+0.3388, +0.4967] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 7 | +0.4551 | [+0.3509, +0.4985] | 100/100 | 7.89e-31 | ✓ |
| sigma_mle_sqrt | 8 | +0.3292 | [+0.1878, +0.4126] | 98/100 | 3.98e-27 | ✓ |

Δ = test_loss(init) − test_loss(true) per member; positive Δ means the trained representation produces lower test error.
