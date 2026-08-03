#!/usr/bin/env bash
# Known-answer tests: every vector in test/vectors is scrambled with each
# implementation and compared against the Python/spec reference output.
set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$DIR")"
BIN="$ROOT/pusch_scrambler"

if [ ! -x "$BIN" ]; then
    echo "error: build the binary first (make)" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0

# Compare payload bytes only: header comments and whitespace layout carry no
# test meaning.
norm() { grep -v '^[[:space:]]*#' "$1" | tr -s '[:space:]' '\n' | grep -v '^$'; }

for in_file in "$DIR"/vectors/*.in; do
    name="$(basename "$in_file" .in)"
    ref="$DIR/vectors/$name.ref"
    if [ ! -f "$ref" ]; then
        echo "SKIP $name (no .ref)"
        continue
    fi

    # Parameters live in the vector's own header comment.
    rnti="$(sed -n 's/.*rnti=\([0-9]*\).*/\1/p' "$in_file" | head -1)"
    nid="$(sed -n 's/.*nid=\([0-9]*\).*/\1/p' "$in_file" | head -1)"
    if [ -z "$rnti" ] || [ -z "$nid" ]; then
        echo "FAIL $name: no rnti/nid header"
        fail=$((fail + 1))
        continue
    fi

    for impl in scalar wordwise avx2; do
        out="$TMP/$name.$impl.out"
        if ! "$BIN" scramble --in "$in_file" --out "$out" \
                --rnti "$rnti" --nid "$nid" --impl "$impl" >/dev/null; then
            echo "FAIL $name [$impl]: binary returned non-zero"
            fail=$((fail + 1))
            continue
        fi

        if diff -q <(norm "$out") <(norm "$ref") >/dev/null; then
            pass=$((pass + 1))
        else
            echo "FAIL $name [$impl]: output differs from reference"
            fail=$((fail + 1))
            continue
        fi

        # Scrambling twice must return the original payload.
        back="$TMP/$name.$impl.back"
        "$BIN" scramble --in "$out" --out "$back" \
            --rnti "$rnti" --nid "$nid" --impl "$impl" >/dev/null
        if diff -q <(norm "$back") <(norm "$in_file") >/dev/null; then
            pass=$((pass + 1))
        else
            echo "FAIL $name [$impl]: descrambling did not recover the input"
            fail=$((fail + 1))
        fi
    done
done

echo "----"
if [ "$fail" -eq 0 ]; then
    echo "PASS: $pass checks"
    exit 0
fi
echo "FAILED: $fail of $((pass + fail)) checks"
exit 1
