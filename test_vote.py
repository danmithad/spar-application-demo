"""Edge cases for majority_vote and prf -- ties, abstentions and absent classes
are where voting and macro-averaging respectively go wrong."""
from judge_eval import majority_vote as mv, prf

# strict majority: 6-of-10 in, 5-of-10 (exact tie) out
assert mv([["risk"]] * 6 + [[]] * 4) == ["risk"]
assert mv([["risk"]] * 5 + [[]] * 5) == [], "exact tie must not include the label"
assert mv([["risk"]] * 5 + [[]] * 6) == []

# odd n has no tie state
assert mv([["risk"]] * 6 + [[]] * 5) == ["risk"]
assert mv([["risk"]] * 5 + [[]] * 6) == []

# labels vote independently -- a passage can carry two labels at once
assert mv([["risk", "vendor"]] * 7 + [["risk"]] * 3) == ["risk", "vendor"]
assert mv([["risk", "vendor"]] * 3 + [["risk"]] * 7) == ["risk"]

# unparsed samples abstain: they shrink the denominator, not vote empty
assert mv([["risk"]] * 3 + [None] * 7) == ["risk"], "abstentions must not outvote"
assert mv([None] * 5) == []
assert mv([]) == []

# empty set is a legitimate prediction
assert mv([[]] * 10) == []

# threshold is configurable (unanimity)
assert mv([["risk"]] * 9 + [[]], threshold=0.99) == []
assert mv([["risk"]] * 10, threshold=0.99) == ["risk"]

# n=1 must be identity -- backwards compatible with the un-voted bench
for s in ([["risk", "adoption"]], [[]], [["harm"]]):
    assert mv(s) == sorted(s[0])

# output is sorted and deduped to canonical label order
assert mv([["vendor", "adoption"]] * 3) == ["adoption", "vendor"]


# ---- prf: a class with no gold positives must not score a vacuous 1.0 ----
# This is the bootstrap bug: 13% of resamples contain no `harm` passage (2 gold
# in 50), and scoring the absent class 1.0 pinned 25% of macro to a constant.
g, p = [["risk"], ["risk"]], [["risk"], ["risk"]]
s = prf(g, p, 1.0)
assert s["macro_support"] == ["risk"], "only gold-supported classes are averaged"
assert s["macro_f"] == 1.0
# and the absent classes must not drag macro down either -- excluded, not zeroed
assert prf([["risk"]], [[]], 1.0)["macro_f"] == 0.0, "sole supported class missed"

# a perfect judge scores 1.0 whether or not every class is represented
assert prf([["risk"], ["adoption"]], [["risk"], ["adoption"]], 1.0)["macro_f"] == 1.0

# support depends on GOLD only, never on predictions -- otherwise two judges in a
# paired bootstrap would average over different denominators and the delta is junk
strict, loose = prf(g, [[], []], 1.0), prf(g, [["risk", "harm"], ["risk"]], 1.0)
assert strict["macro_support"] == loose["macro_support"] == ["risk"]
# false positives on an absent class still show up in micro, just not in macro
assert loose["micro_p"] < 1.0 and loose["macro_f"] == 1.0

# no gold labels anywhere: macro falls back to micro, which is 1.0 iff the judge
# also predicted nothing
assert prf([[], []], [[], []], 1.0)["macro_f"] == 1.0
assert prf([[], []], [["risk"], []], 1.0)["macro_f"] == 0.0

# beta still moves precision/recall weighting after the change
assert prf([["risk"]], [["risk", "adoption"]], 0.25)["micro_f"] < \
       prf([["risk"]], [["risk", "adoption"]], 4.0)["micro_f"]


# ---- paired_bootstrap: cluster resampling ----
from judge_eval import paired_bootstrap as pb, clusters_of

G = [["risk"], ["risk"], [], ["adoption"], ["adoption"], [], ["risk"], []]
A = [["risk"], ["risk"], [], ["adoption"], [], [], ["risk"], []]        # 1 miss
C = [["risk"], [], [], ["adoption"], [], ["risk"], ["risk"], ["harm"]]  # 3 errors

# singleton clusters must be EXACTLY the old passage-level bootstrap: same rng
# draws, same indices. This is what makes clusters=None a safe default.
assert pb(G, A, C, 1.0, B=300) == {**pb(G, A, C, 1.0, B=300,
                                        clusters=list(range(len(G)))),
                                   "n_clusters": len(G)}

# one cluster containing everything -> every resample is the identical dataset,
# so the delta never varies and the interval collapses
one = pb(G, A, C, 1.0, B=200, clusters=[0] * len(G))
assert one["lo"] == one["hi"], "a single cluster cannot produce spread"
assert one["n_clusters"] == 1

# cluster ids are reported, and derived from the passage field
assert pb(G, A, C, 1.0, B=100, clusters=[0, 0, 0, 0, 1, 1, 1, 1])["n_clusters"] == 2
assert clusters_of([{"company": "X"}, {"company": "Y"}, {"company": "X"}]) \
       == ["X", "Y", "X"]

# clustering must not be able to NARROW the interval on clustered data: errors
# concentrated in whole clusters are what passage resampling breaks apart
gold = [["risk"]] * 12
good = [["risk"]] * 12
bad = [["risk"]] * 6 + [[]] * 6           # all errors inside clusters 2 and 3
grp = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
flat_w = (lambda r: r["hi"] - r["lo"])(pb(gold, good, bad, 1.0, B=800))
clus_w = (lambda r: r["hi"] - r["lo"])(pb(gold, good, bad, 1.0, B=800, clusters=grp))
assert clus_w >= flat_w, f"clustered CI narrower than passage CI ({clus_w} < {flat_w})"

# pairing survives clustering: a judge compared against itself has delta==0 always
self_cmp = pb(G, A, A, 1.0, B=200, clusters=[0, 0, 1, 1, 2, 2, 3, 3])
assert self_cmp["lo"] == self_cmp["hi"] == 0.0, "pairing broken under clustering"

print("all majority_vote, prf and cluster-bootstrap edge cases pass")
