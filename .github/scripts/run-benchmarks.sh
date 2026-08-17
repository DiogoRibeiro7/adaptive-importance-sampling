#!/usr/bin/env bash
# Run the benchmark suite and write results for one side of a comparison.
#
# Usage: run-benchmarks.sh <pr|base>
#
# Both sides of the comparison call this, so the only difference between them is
# which commit is checked out. .benchmarks/ is gitignored, so results written
# here survive the `git checkout` between the two runs.
set -euo pipefail

side="${1:?expected 'pr' or 'base'}"
out=".benchmarks/${side}_results.json"

mkdir -p .benchmarks

if compgen -G "benchmarks/test_*.py" > /dev/null; then
  python -m pytest benchmarks/ --benchmark-only --benchmark-json="$out" || true
fi

# Always leave a readable file behind. upload-artifact creates nothing for an
# empty directory, and the compare step would then fail on a missing artifact
# rather than reporting that there was no baseline. A base commit predating a
# benchmark legitimately has nothing to report.
if [ ! -f "$out" ]; then
  echo '{"benchmarks": []}' > "$out"
fi

echo "${side} benchmarks written to ${out}"
