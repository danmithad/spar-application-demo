# Who Grades the Graders?

A small end-to-end demo of LLM-judge validation methodology, built around the
question: *if an LLM judge assigns more labels than human annotators, and spot
checks suggest the extra labels are valid, is the LLM judge better?*

It runs on real data from the [AI Risk Observatory](https://github.com/84rt/AI-Risk-Observatory)
(AISI, CC BY 4.0): passages from UK public-company annual reports, multi-labeled
as `adoption / risk / vendor / harm`.

**Live site:** interactive results — an F_beta dial and micro/macro toggle that
recompute the model ranking and paired-bootstrap CIs client-side.

## The recipe

1. **Ground truth** — 50 passages sampled from the AIRO golden set (stratified
   so all four labels and the no-label case appear; all 3 passages the AIRO
   pipeline flagged `harm` in the 24k corpus included — the rubric then demoted
   one of them, p50, to `risk`, so **2** passages carry gold `harm`),
   hand-labeled against a written rubric
   ([RUBRIC.md](RUBRIC.md)) fixed before any judge ran. Labeler: Claude
   (Fable 5) — a declared conflict, since Anthropic models are on the bench.
2. **Metric** — per-class and micro/macro F_beta; beta is the explicit
   FP-vs-FN exchange rate rather than a hidden assumption.
3. **Inference** — paired bootstrap (B = 10,000): whole **companies** resampled
   with replacement, every passage of a drawn filer coming with it, keeping gold
   labels and both judges' predictions together; one-sided p = fraction of
   resamples with Δ ≤ 0. Passages from one filer are not independent draws —
   risk boilerplate is copied forward and judge errors repeat across a document —
   and the unit sets the estimand: resampling companies asks how a judge would do
   on filers it has not seen, which is the question a prevalence observatory asks.

## Findings (n = 50, beta = 1, macro)

Each judge is scored by majority vote over 30 independent samples per passage,
which estimates its modal answer rather than one lucky draw.

| judge | macro-F1 | micro P | micro R |
|---|---|---|---|
| gpt-5 | 0.935 | 0.911 | 0.932 |
| claude-opus-5 | 0.933 | 0.894 | 0.955 |
| claude-sonnet-5 | 0.914 | 0.796 | 0.977 |
| gpt-5-mini | 0.901 | 0.782 | 0.977 |
| gpt-5-nano | 0.876 | 0.784 | 0.909 |
| claude-haiku-4-5 | 0.813 | 0.717 | 0.864 |

- The top four are statistically indistinguishable at this n (p = 0.07–0.45).
  The #1–#2 gap is 0.002 against a 95% CI of ±0.092.
- Five of six judges over-label (recall ≥ precision) — the phenomenon in the
  motivating question, reproduced.
- Aggregation alone flips a rank: sonnet beats gpt-5-mini on macro, ties on micro.

## Does self-consistency fix the unstable leaderboard?

Short answer: it stabilises the *scores* and changes nothing that matters.
`stability.py` runs the experiment against a 30-sample-per-passage pool
(`--samples 30`), including three fully disjoint n=10 runs.

| | n = 1 | n = 10 majority vote |
|---|---|---|
| run-to-run sd of macro-F1 | 0.0145 | 0.0051 |
| distinct rankings observed | 12 (over 30 runs) | 2 (over 3 runs) |
| judges separated from #1 at p<0.05 | 2 of 5 | 2 of 5 |

- Voting removes **65%** of judge sampling noise, but the eval set's own
  bootstrap CI (width 0.197) is **38× wider** than what remains. The binding
  constraint is 50 passages, not sampling.
- **The #1 spot still flips** at n=10 (2 runs put opus first, 1 put gpt-5 first),
  because the true gap between them is ~0.002.
- Voting barely helps accuracy: mean macro-F1 gain from n=1 to unlimited voting
  is **+0.004**, and negative for three of six judges. Across the bench, 63
  passages have a *wrong modal answer* (bias, which voting cannot fix) against
  only 8 that are merely shaky (variance, which it can). Ten samples is a 10×
  bill for the wrong failure mode.
- Resolving the top gap by bootstrap would need on the order of **350,000
  passages**. More labelled data, not more votes.
- Voting also *hides* ambiguity: 15 passages are contested by multiple judges
  (gpt-5-mini splits p38 four ways, 10/9/6/5). A majority vote reports those as
  confident answers, moving the instability out of the score and into the rubric.

## Two harness bugs worth flagging

### 1. Truncation scored as a confident empty answer

The original bench capped Anthropic responses at `max_tokens: 200`. claude-opus-5
spends ~430 output tokens, nearly all of them thinking, so it was truncated
mid-thought on **11.9%** of calls — returning no text, failing to parse, and
being silently scored as *"predicted no labels."* The OpenAI path set no cap, so
this penalised Anthropic judges only.

Fixing it (cap 2000, and truncation now raises instead of returning an empty
prediction) moved opus-5 from macro-F1 0.898 to 0.927, cut **29%** of its
run-to-run variance, and took it from placing first in 1 of 30 single-sample
runs to 9 of 30. A meaningful share of what looked like "LLM judge stochasticity"
was the harness quietly scoring truncations as wrong answers.

### 2. Macro-averaging handed out free points, and the bootstrap kept collecting them

`prf` scored a class with no gold positives as P = R = 1 → **F = 1.0**, the
vacuous case. On the full 50 passages every class is represented, so the
leaderboard was unaffected — but the paired bootstrap resamples passages, and
`harm` has only 2 gold instances. Measured: **1,314 of 10,000 resamples contain
no `harm` passage at all**. In each one, 25% of macro-F1 became the constant 1.0
for *both* judges, scaling that resample's paired delta by 3/4.

Because the vacuous 1.0 was awarded to both judges symmetrically, the sign of
each delta was preserved — so **p-values are unchanged** — but the spread was
compressed and every confidence interval came out too narrow:

| Δ vs gpt-5 | CI width before | after |
|---|---|---|
| claude-opus-5 | 0.1409 | 0.1518 |
| claude-sonnet-5 | 0.0823 | 0.0879 |
| gpt-5-mini | 0.0976 | 0.1072 |
| gpt-5-nano | 0.1036 | 0.1146 |
| claude-haiku-4-5 | 0.1529 | 0.1645 |

(Both columns are passage-level, isolating this one fix; the bootstrap now
resamples companies, which widens them further.)

macro now averages only over classes with gold support, and the support set is a
function of the gold labels alone — never the predictions — so both judges in a
paired comparison are always averaged over the same denominator. False positives
on an absent class still surface in micro. The same fix is mirrored in the site's
client-side `prf`, which had the identical bug and drives the interactive CIs.

The lesson is the same one the page is about: the reported uncertainty was
6–11% too confident, and nothing in the leaderboard would have revealed it.

## Layout

```
RUBRIC.md          the hand-written ground-truth criterion
judge_eval.py      judges + majority vote + metrics + paired bootstrap (stdlib only)
stability.py       does n-sample voting fix the ranking? (reads the pool, no API calls)
impact.py          truncation-bug impact + how many passages would separate the top
rebuild_results.py re-score results.json from the pool, no API calls
test_vote.py       majority_vote + prf edge cases (ties, abstentions, absent classes)
build_site.py      injects data/results into the site template
run.sh             end-to-end: judge all models, rebuild the site
data/passages.jsonl        the 50 gold-labeled passages
results/results.json       per-passage predictions per model
results/samples_pool.jsonl raw per-sample labels (30 per judge per passage)
results/stability.json     the self-consistency experiment's numbers
site/              the static, self-contained results page
```

## Run it

```
printf 'ANTHROPIC_API_KEY=...\nOPENAI_API_KEY=...\n' > .env
./run.sh
```

`run.sh` defaults to `SAMPLES=30`, which is exactly what produced the committed
`results/results.json`, the sample pool, and the table above — so running it
reproduces this README rather than quietly replacing it with a different
experiment. That is ~9,000 calls (~45 min, roughly $15–20).

For a cheap look, `SAMPLES=1 ./run.sh` is ~300 calls (<$2). It writes to
`results/results_n1.json` / `results/samples_n1.jsonl` and skips the site build,
so it cannot clobber the committed artifacts.

Re-analysis is free — it reads the sample pool, never the API:

```
python3 stability.py        # the self-consistency experiment
python3 impact.py           # bug impact + passages needed to separate the top
python3 rebuild_results.py  # re-score results.json from the pool
python3 test_vote.py        # majority_vote + prf edge cases
```

The raw AIRO dataset (not committed here; ~380 MB) is only needed to re-draw the
sample — the 50 sampled passages are committed.

## Limitations

Single (machine) annotator with no inter-annotator agreement ceiling;
judge–adjudicator dependence (Anthropic gold, Anthropic judges); sampling
stratified on the pipeline's own phase-1 labels; single un-tuned operating
point per judge; only 44 companies to resample, which is near the lower bound
where a cluster bootstrap stays honest.
The site's Item 7 spells each of these out.

---

Data: AI Risk Observatory Dataset v1.0, AI Security Institute, April 2026,
CC BY 4.0. Built as a methodology demo for a SPAR application.
