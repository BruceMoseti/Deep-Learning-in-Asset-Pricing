#!/usr/bin/env bash
# Full pipeline, in dependency order.  Equivalent to `make all`; kept as a
# script so it can be launched detached and its log tailed.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

for stage in \
    00_fetch_data.py \
    01_model_comparison.py \
    02_portfolios.py \
    03_inference.py \
    04_high_dimensional.py \
    05_robustness.py \
    06_figures.py
do
    echo "=============== ${stage} ==============="
    started=$SECONDS
    python3 -u "scripts/${stage}" "$@"
    echo "  ${stage} took $((SECONDS - started))s"
done
echo "=============== done ==============="
