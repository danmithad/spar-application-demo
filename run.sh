#!/bin/bash
# Runs the judge bench end-to-end. Needs .env with ANTHROPIC_API_KEY and OPENAI_API_KEY.
# judge_eval.py reads .env itself, so no shell sourcing is required.
#
# SAMPLES = independent judgements per passage. The default of 30 is what produced
# the committed results and the README's table: each judge is scored by majority
# vote over 30 draws, estimating its modal answer rather than one lucky sample.
# That is ~9,000 API calls (~45 min, roughly $15-20).
#
# For a cheap look, SAMPLES=1 is ~300 calls (<$2). It writes to its own files and
# does not rebuild the site, so it cannot overwrite the committed n=30 artifacts.
set -e
cd "$(dirname "$0")"

CANON=30
SAMPLES="${SAMPLES:-$CANON}"
MODELS="anthropic:claude-haiku-4-5,anthropic:claude-sonnet-5,anthropic:claude-opus-5,openai:gpt-5-nano,openai:gpt-5-mini,openai:gpt-5"

if [ "$SAMPLES" = "$CANON" ]; then
  POOL=results/samples_pool.jsonl
  RESULTS=results/results.json
else
  POOL="results/samples_n${SAMPLES}.jsonl"
  RESULTS="results/results_n${SAMPLES}.json"
  echo "SAMPLES=$SAMPLES is not the canonical $CANON: writing $RESULTS / $POOL"
  echo "and leaving the committed results.json, pool, and site untouched."
fi

python3 judge_eval.py --models "$MODELS" --samples "$SAMPLES" \
  --samples-out "$POOL" --results-out "$RESULTS"

if [ "$SAMPLES" = "$CANON" ]; then
  python3 build_site.py "$RESULTS"
  echo "done — site/index.html ready to republish"
  echo "re-run the free analyses over the new pool with:"
  echo "  python3 stability.py && python3 impact.py"
else
  echo "done — wrote $RESULTS and $POOL (site not rebuilt)"
  echo "to view it anyway: python3 build_site.py $RESULTS"
fi
