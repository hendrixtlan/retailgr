#!/usr/bin/env bash
# Fetch the MovieLens datasets used by the model experiments.
#
# MovieLens has no item variants, so it does not answer the granularity
# question. It answers the other one: with real timestamps and an explicit
# rating per interaction, it is where HSTU's temporal bias and action modality
# have genuine signal to read.
#
# Both files come from public GitHub mirrors rather than files.grouplens.org,
# so this works from networks that only allow GitHub. The canonical source is
# https://grouplens.org/datasets/movielens/ - please honour its terms of use.

set -euo pipefail

RAW_DIR="${1:-data/raw}"
NCF_BASE="https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/master/Data"
RECBOLE_BASE="https://raw.githubusercontent.com/RUCAIBox/RecBole/master/dataset/ml-100k"

echo "==> MovieLens-1M (1,000,209 ratings, 6,040 users)"
mkdir -p "$RAW_DIR/ml1m"
for part in train test; do
    target="$RAW_DIR/ml1m/ml-1m.$part.rating"
    if [ -s "$target" ]; then
        echo "    have $target"
    else
        curl -sSLf -o "$target" "$NCF_BASE/ml-1m.$part.rating"
        echo "    fetched $target ($(wc -l < "$target") rows)"
    fi
done

echo "==> MovieLens-100K (100,000 ratings, with genres)"
mkdir -p "$RAW_DIR/ml100k"
for name in ml-100k.inter ml-100k.item; do
    target="$RAW_DIR/ml100k/$name"
    if [ -s "$target" ]; then
        echo "    have $target"
    else
        curl -sSLf -o "$target" "$RECBOLE_BASE/$name"
        echo "    fetched $target ($(wc -l < "$target") rows)"
    fi
done

cat <<'EOF'

Ready. Next:

    retailgr experiment --dataset ml1m --variants config
    retailgr ablate     --dataset ml1m

Only the `config` variant is worth running on MovieLens: with no item
variants, every granularity level resolves to the same tokens.
EOF
