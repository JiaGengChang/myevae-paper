#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
output_file="${1:-$project_root/nan_counts.csv}"
cd "$project_root"

if [[ -x "$project_root/.venv/bin/python" ]]; then
    python_executable="$project_root/.venv/bin/python"
else
    python_executable="${PYTHON:-python3}"
fi

counter_script="$script_dir/22_count_nans.py"

printf 'endpoint,shuffle,fold,nan_count,total_count,pcMissing\n' > "$output_file"

for endpoint in os pfs; do
    for shuffle in {0..9}; do
        for fold in {0..4}; do
            output="$($python_executable "$counter_script" \
                --endpoint "$endpoint" \
                --shuffle "$shuffle" \
                --fold "$fold")"

            total_count="$(printf '%s\n' "$output" | sed -n '1p')"
            nan_count="$(printf '%s\n' "$output" | sed -n '2p')"
            pc_missing="$(printf '%s\n' "$output" | sed -n '3p' | sed 's/^%Missing: *//')"

            if [[ -z "$total_count" || -z "$nan_count" || -z "$pc_missing" ]]; then
                echo "Unable to parse output for endpoint=$endpoint shuffle=$shuffle fold=$fold" >&2
                exit 1
            fi

            printf '%s,%s,%s,%s,%s,%s\n' \
                "$endpoint" "$shuffle" "$fold" "$nan_count" "$total_count" "$pc_missing" \
                >> "$output_file"
        done
    done
done

printf 'Wrote %s\n' "$output_file"
