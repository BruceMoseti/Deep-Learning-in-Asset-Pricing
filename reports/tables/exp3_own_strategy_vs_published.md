| model         |   tstat_nw |   percentile_among_published | passes_uncorrected   | passes_bonferroni   | passes_tstat_3_hurdle   |   ff6_alpha_tstat | ff6_alpha_passes_bonferroni   |
|:--------------|-----------:|-----------------------------:|:---------------------|:--------------------|:------------------------|------------------:|:------------------------------|
| single-signal |     2.0343 |                      25.0000 | True                 | False               | False                   |            2.1261 | False                         |
| ols           |     2.4297 |                      34.4340 | True                 | False               | False                   |            1.5683 | False                         |
| ridge         |     2.4305 |                      34.4340 | True                 | False               | False                   |            1.5688 | False                         |
| lasso         |     3.0957 |                      50.4717 | True                 | False               | True                    |            2.6225 | False                         |
| enet          |     3.1594 |                      50.4717 | True                 | False               | True                    |            2.8044 | False                         |
| xgboost       |     3.4292 |                      57.0755 | True                 | False               | True                    |            2.8340 | False                         |
| neural-net    |     2.2532 |                      30.6604 | True                 | False               | False                   |            1.7759 | False                         |
