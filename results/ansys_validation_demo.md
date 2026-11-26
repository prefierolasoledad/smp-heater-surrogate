# ANSYS validation (DEMO: FD stands in for ANSYS)

Runs used: 25

| Predictor of ANSYS band activation time | Leave-one-out MAPE |
|---|---|
| lumped equations alone | 13.8 % |
| lumped x constant ANSYS ratio | 5.5 % |
| Delta-XGBoost (lumped + ML correction) | 4.6 % |

| Ratio | Median | Min | Max |
|---|---|---|---|
| ratio_ansys_to_lumped_band | 1.162 | 1.048 | 1.353 |
| ratio_ansys_to_fd | 1.000 | 1.000 | 1.000 |

baseline: ANSYS 32.2 s · 2D FD 32.2 s · lumped band 30.0 s

optimum D: ANSYS 22.6 s · 2D FD 22.6 s · lumped band 20.5 s

Speed-up of D over baseline in ANSYS: 1.42x (lumped: 1.45x, 2D FD: 1.42x)
