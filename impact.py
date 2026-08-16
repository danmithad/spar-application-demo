"""Two closing questions.

1. How much of the original run-to-run instability was the max_tokens truncation
   bug rather than genuine judge sampling noise?
2. Given voting cannot separate the top judges, how many PASSAGES would?
"""
import collections
import json
import math
import statistics

from judge_eval import majority_vote, prf, paired_bootstrap, clusters_of
from stability import load_pool, score, ranking_of

BETA = 1.0
CLUSTER_KEY = "company"
SHORT = lambda m: m.split(":", 1)[1]

passages, golds, fixed = load_pool("results/samples_pool.jsonl", "data/passages.jsonl")
_, _, dirty = load_pool("results/samples_pool_contaminated.jsonl", "data/passages.jsonl")

print("=" * 74)
print("1. TRUNCATION BUG: how much of opus's instability was the harness?")
print("=" * 74)
M = "anthropic:claude-opus-5"
print(f"  {'opus-5 pool':22}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}{'spread':>9}"
      f"{'unparsed':>10}")
for name, pool in (("before fix (cap 200)", dirty), ("after fix (cap 2000)", fixed)):
    vals = [prf(golds, [majority_vote(s[i:i + 1]) for s in pool[M]], BETA)["macro_f"]
            for i in range(30)]
    bad = sum(1 for s in pool[M] for x in s if x is None)
    print(f"  {name:22}{statistics.mean(vals):>8.3f}{statistics.stdev(vals):>8.3f}"
          f"{min(vals):>8.3f}{max(vals):>8.3f}{max(vals)-min(vals):>9.3f}{bad:>10}")

sd_dirty = statistics.stdev(
    [prf(golds, [majority_vote(s[i:i+1]) for s in dirty[M]], BETA)["macro_f"]
     for i in range(30)])
sd_clean = statistics.stdev(
    [prf(golds, [majority_vote(s[i:i+1]) for s in fixed[M]], BETA)["macro_f"]
     for i in range(30)])
print(f"\n  opus n=1 run-to-run sd: {sd_dirty:.4f} -> {sd_clean:.4f} "
      f"({(1-sd_clean/sd_dirty)*100:.0f}% of it was the truncation bug, not the judge)")

# how often did the bug alone change who placed #1?
models = list(fixed)
wins = {}
for name, pool in (("before fix", dirty), ("after fix", fixed)):
    c = collections.Counter()
    for i in range(30):
        sc = {m: prf(golds, [majority_vote(s[i:i+1]) for s in pool[m]], BETA)["macro_f"]
              for m in models}
        c[ranking_of(sc)[0]] += 1
    wins[name] = c
    print(f"  {name:12} #1 placements over 30 single-sample runs: "
          + ", ".join(f"{SHORT(m)} {n}" for m, n in c.most_common()))

print()
print("=" * 74)
print("2. WHAT WOULD ACTUALLY SEPARATE THE TOP JUDGES? (more passages, not votes)")
print("=" * 74)
# vote the full pool = best available estimate of each judge's true score
final = {m: score(golds, fixed[m], BETA)[0] for m in models}
ranked = ranking_of(final)
top, second = ranked[0], ranked[1]
preds = {m: [majority_vote(s) for s in fixed[m]] for m in models}
cl = clusters_of(passages, CLUSTER_KEY)
bs = paired_bootstrap(golds, preds[top], preds[second], BETA, clusters=cl)
gap = final[top] - final[second]
width = bs["hi"] - bs["lo"]
print(f"  best estimate of the top gap: {SHORT(top)} - {SHORT(second)} = {gap:+.4f}")
print(f"  95% CI at n={len(passages)} passages: [{bs['lo']:+.3f}, {bs['hi']:+.3f}]"
      f"  width {width:.3f}  p(d<=0)={bs['p_leq_0']:.3f}")
# CI width shrinks ~1/sqrt(n); solve for the n where width ~= the gap
if gap > 0:
    need = len(passages) * (width / gap) ** 2
    print(f"\n  CI width scales ~1/sqrt(n_passages), so resolving a {gap:.4f} gap needs")
    print(f"  roughly {need:,.0f} passages -- vs the {len(passages)} you have.")
    print(f"  (~{need/len(passages):,.0f}x more labelling, and no amount of voting substitutes.)")

print()
print("  gaps between adjacent judges, best estimate:")
for a, b in zip(ranked, ranked[1:]):
    print(f"    {SHORT(a):18} - {SHORT(b):18} = {final[a]-final[b]:+.4f}")

json.dump({"final_scores": final, "ranking": ranked, "top_gap": gap,
           "top_ci": bs, "passages_needed": need if gap > 0 else None,
           "opus_sd_before": sd_dirty, "opus_sd_after": sd_clean,
           "cluster_key": CLUSTER_KEY},
          open("results/impact.json", "w"), indent=1)
print("\nwrote results/impact.json")
