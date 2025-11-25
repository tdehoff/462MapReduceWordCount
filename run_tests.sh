#!/usr/bin/env bash

TEST_DIR="raw_text_input"

args=()

for file in "$TEST_DIR"/*; do
    args+=("$file")
done

./seq "${args[@]}" > seq_out.txt
./omp "${args[@]}" > omp_out.txt
diff seq_out.txt omp_out.txt