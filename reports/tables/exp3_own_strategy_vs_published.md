| model         |   tstat_nw |   percentile_among_published | passes_uncorrected   | passes_bonferroni   | passes_tstat_3_hurdle   |   ff6_alpha_tstat | ff6_alpha_passes_bonferroni   |
|:--------------|-----------:|-----------------------------:|:---------------------|:--------------------|:------------------------|------------------:|:------------------------------|
| single-signal |     2.0343 |                      25.0000 | True                 | False               | False                   |            2.1261 | False                         |
| ols           |     2.4297 |                      34.4340 | True                 | False               | False                   |            1.5683 | False                         |
| ridge         |     2.4460 |                      35.3774 | True                 | False               | False                   |            1.5846 | False                         |
| lasso         |     1.9527 |                      23.1132 | False                | False               | False                   |            1.5815 | False                         |
| enet          |     3.0053 |                      48.5849 | True                 | False               | True                    |            2.6477 | False                         |
| xgboost       |     3.4390 |                      57.0755 | True                 | False               | True                    |            2.8337 | False                         |
| neural-net    |     2.2918 |                      31.6038 | True                 | False               | False                   |            1.8066 | False                         |
