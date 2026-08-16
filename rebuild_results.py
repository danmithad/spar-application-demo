"""Regenerate results/results.json from the corrected sample pool.

Point estimate = majority vote over the whole 30-sample pool, which is the best
available estimate of each judge's modal answer. Per-judge sampling spread and
the n=10 reproducibility check live in results/stability.json.
"""
import json

from judge_eval import (LABELS, majority_vote, prf, paired_bootstrap, clusters_of)
from stability import load_pool, ranking_of

BETA = 1.0
passages, golds, pool = load_pool("results/samples_pool.jsonl", "data/passages.jsonl")
raw = [json.loads(l) for l in open("data/passages.jsonl") if l.strip()]

CLUSTER_KEY = "company"
out = {"labels": LABELS, "beta": BETA, "samples": 30, "vote_threshold": 0.5,
       "cluster_key": CLUSTER_KEY,
       "passages": [{"id": p["id"], "text": p["text"], "gold": sorted(p["gold"])}
                    for p in raw],
       "models": {}}

for m, samples in pool.items():
    preds = [majority_vote(s) for s in samples]
    out["models"][m] = {"preds": preds, "stats": prf(golds, preds, BETA)}

ranked = ranking_of({m: out["models"][m]["stats"]["macro_f"] for m in out["models"]})
out["ranking"] = ranked
top = ranked[0]
cl = clusters_of(passages, CLUSTER_KEY)
out["bootstrap_vs_top"] = {
    m: paired_bootstrap(golds, out["models"][top]["preds"],
                        out["models"][m]["preds"], BETA, clusters=cl)
    for m in ranked[1:]}

with open("results/results.json", "w") as f:
    json.dump(out, f, indent=1)

print("ranking:", " > ".join(m.split(':')[1] for m in ranked))
for m in ranked:
    s = out["models"][m]["stats"]
    print(f"  {m.split(':')[1]:20} macro-F={s['macro_f']:.3f} "
          f"micro-F={s['micro_f']:.3f} P={s['micro_p']:.3f} R={s['micro_r']:.3f}")
print(f"\nbootstrap resampling unit: {CLUSTER_KEY} "
      f"({len(set(cl))} clusters over {len(passages)} passages)")
print("\nwrote results/results.json")
