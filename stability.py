"""Does self-consistency (n-sample majority vote) fix the unstable leaderboard?

Reads the raw sample pool written by judge_eval.py --samples K and answers three
separate questions that are easy to conflate:

  Q1 REPRODUCIBILITY  Do two independent runs of the same bench agree?
                      Driven by judge sampling noise. Majority vote attacks this.
  Q2 SIGNIFICANCE     Is the #1-vs-#2 gap bigger than the eval set can resolve?
                      Driven by having only 50 passages. Majority vote CANNOT
                      touch this -- voting changes the judge, not the ruler.
  Q3 ESTIMAND         n=1 and n=10 score different systems. A vote of 10 is a
                      10x-cost judge; ranking it does not rank the 1-call judge.

Usage:  python3 stability.py [--pool results/samples_pool.jsonl] [--beta 1.0]
"""

import argparse
import collections
import itertools
import json
import random
import statistics

from judge_eval import (LABELS, majority_vote, prf, paired_bootstrap, clusters_of)

SHORT = lambda m: m.split(":", 1)[1]


def load_pool(pool_path, passages_path):
    passages = [json.loads(l) for l in open(passages_path) if l.strip()]
    order = {p["id"]: i for i, p in enumerate(passages)}
    golds = [sorted(p["gold"]) for p in passages]

    pool = collections.defaultdict(lambda: [None] * len(passages))
    for line in open(pool_path):
        if not line.strip():
            continue
        d = json.loads(line)
        pool[d["model"]][order[d["id"]]] = d["samples"]
    return passages, golds, {m: v for m, v in pool.items()}


def score(golds, samples_per_passage, beta, threshold=0.5):
    preds = [majority_vote(s, threshold) for s in samples_per_passage]
    return prf(golds, preds, beta)["macro_f"], preds


def kendall_distance(rank_a, rank_b):
    """Fraction of model pairs ordered differently by two rankings."""
    pos_a = {m: i for i, m in enumerate(rank_a)}
    pos_b = {m: i for i, m in enumerate(rank_b)}
    pairs = list(itertools.combinations(rank_a, 2))
    disc = sum(1 for x, y in pairs
               if (pos_a[x] < pos_a[y]) != (pos_b[x] < pos_b[y]))
    return disc / len(pairs)


def ranking_of(scores):
    # tie-break by name: an exact tie must not rank by dict insertion order
    return sorted(scores, key=lambda m: (-scores[m], m))


def disjoint_runs(models, pool, golds, beta, n, k_pool):
    """Genuinely independent benches: run i uses samples [i*n:(i+1)*n], no reuse."""
    runs = []
    for i in range(k_pool // n):
        sl = slice(i * n, (i + 1) * n)
        scores, preds = {}, {}
        for m in models:
            scores[m], preds[m] = score(golds, [s[sl] for s in pool[m]], beta)
        runs.append({"scores": scores, "preds": preds, "ranking": ranking_of(scores)})
    return runs


def resampled_runs(models, pool, golds, beta, n, reps, rng):
    """Simulate `reps` n-sample benches by drawing n of the K pooled samples."""
    out = []
    k = len(pool[models[0]][0])
    for _ in range(reps):
        idx = rng.sample(range(k), n)
        scores = {m: score(golds, [[s[i] for i in idx] for s in pool[m]], beta)[0]
                  for m in models}
        out.append(scores)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="results/samples_pool.jsonl")
    ap.add_argument("--passages", default="data/passages.jsonl")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="results/stability.json")
    ap.add_argument("--cluster-key", default="company",
                    help="bootstrap resampling unit ('none' = passage-level)")
    args = ap.parse_args()
    rng = random.Random(0)

    passages, golds, pool = load_pool(args.pool, args.passages)
    models = sorted(pool, key=lambda m: -score(golds, pool[m], args.beta)[0])
    k_pool = len(pool[models[0]][0])
    report = {"k_pool": k_pool, "n_passages": len(passages), "beta": args.beta,
              "cluster_key": args.cluster_key}
    print(f"pool: {len(models)} models x {len(passages)} passages x {k_pool} samples\n")

    # --- judge self-agreement: how noisy is each judge before any voting? -----
    print("=" * 74)
    print("JUDGE SELF-AGREEMENT (over the raw pool, no voting)")
    print("=" * 74)
    print(f"{'judge':22} {'unanimous':>10} {'mean pairwise':>14} {'unparsed':>9}")
    agree = {}
    for m in models:
        unan = sum(1 for s in pool[m]
                   if len({tuple(x) for x in s if x is not None}) == 1)
        # mean pairwise exact-match agreement across the K samples per passage
        pw = []
        for s in pool[m]:
            ok = [tuple(x) for x in s if x is not None]
            if len(ok) > 1:
                pairs = list(itertools.combinations(ok, 2))
                pw.append(sum(1 for a, b in pairs if a == b) / len(pairs))
        bad = sum(1 for s in pool[m] for x in s if x is None)
        agree[m] = {"unanimous": unan / len(passages),
                    "mean_pairwise": statistics.mean(pw) if pw else 1.0,
                    "unparsed": bad}
        print(f"{SHORT(m):22} {unan/len(passages):>9.0%} "
              f"{agree[m]['mean_pairwise']:>13.1%} {bad:>9}")
    report["self_agreement"] = agree

    # --- Q1: three fully independent n=10 runs, exactly as asked --------------
    print()
    print("=" * 74)
    print("Q1  THREE INDEPENDENT n=10 MAJORITY-VOTE RUNS (disjoint samples)")
    print("=" * 74)
    runs10 = disjoint_runs(models, pool, golds, args.beta, 10, k_pool)
    hdr = "".join(f"{'run'+str(i+1):>9}" for i in range(len(runs10)))
    print(f"{'judge':22}{hdr}{'spread':>9}")
    for m in models:
        vals = [r["scores"][m] for r in runs10]
        print(f"{SHORT(m):22}" + "".join(f"{v:>9.3f}" for v in vals)
              + f"{max(vals)-min(vals):>9.3f}")
    print()
    for i, r in enumerate(runs10):
        print(f"  run{i+1} ranking: " + " > ".join(SHORT(m) for m in r["ranking"]))
    same = len({tuple(r["ranking"]) for r in runs10}) == 1
    print(f"\n  all three rankings identical: {same}")
    # a tie sorts by insertion order, so tied ranks masquerade as agreement
    ties = [(i + 1, SHORT(a), SHORT(b))
            for i, r in enumerate(runs10)
            for a, b in itertools.combinations(models, 2)
            if abs(r["scores"][a] - r["scores"][b]) < 1e-12]
    if ties:
        print("  NOTE exact ties (rank order between these is arbitrary):")
        for i, a, b in ties:
            print(f"    run{i}: {a} == {b}")
    report["n10_exact_ties"] = ties
    report["n10_runs"] = [{"scores": r["scores"], "ranking": r["ranking"]}
                          for r in runs10]
    report["n10_rankings_identical"] = same

    # --- the n=1 baseline, same treatment, for contrast -----------------------
    print()
    print("=" * 74)
    print("BASELINE  THIRTY INDEPENDENT n=1 RUNS (one sample each, no voting)")
    print("=" * 74)
    runs1 = disjoint_runs(models, pool, golds, args.beta, 1, k_pool)
    print(f"{'judge':22}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}{'spread':>9}")
    for m in models:
        vals = [r["scores"][m] for r in runs1]
        print(f"{SHORT(m):22}{statistics.mean(vals):>8.3f}"
              f"{statistics.stdev(vals):>8.3f}{min(vals):>8.3f}{max(vals):>8.3f}"
              f"{max(vals)-min(vals):>9.3f}")
    r1 = collections.Counter(tuple(r["ranking"]) for r in runs1)
    print(f"\n  distinct rankings across 30 n=1 runs: {len(r1)}")
    print(f"  most common ranking seen {r1.most_common(1)[0][1]}/30 times")
    winners1 = collections.Counter(r["ranking"][0] for r in runs1)
    print("  who placed #1: " +
          ", ".join(f"{SHORT(m)} {c}/30" for m, c in winners1.most_common()))
    report["n1_runs"] = [{"scores": r["scores"], "ranking": r["ranking"]}
                         for r in runs1]

    # --- Q1b: stability as a function of n -----------------------------------
    print()
    print("=" * 74)
    print("HOW STABILITY SCALES WITH n  (resampled from the pool)")
    print("=" * 74)
    print(f"  caveat: n-subsets are drawn from the same {k_pool}-sample pool, so reps")
    print(f"  overlap and the spread is optimistic as n -> {k_pool}. The unbiased")
    print("  numbers are the disjoint runs above; this curve shows the shape.")
    print(f"{'n':>4}{'mean sd of macro-F':>21}{'P(modal ranking)':>19}"
          f"{'mean pair-disagree':>20}")
    curve = {}
    for n in [1, 3, 5, 7, 11, 15, 21]:
        if n > k_pool:
            continue
        runs = resampled_runs(models, pool, golds, args.beta, n, args.reps, rng)
        sds = [statistics.stdev([r[m] for r in runs]) for m in models]
        ranks = [tuple(ranking_of(r)) for r in runs]
        modal = collections.Counter(ranks).most_common(1)[0]
        kd = statistics.mean(kendall_distance(list(r), list(modal[0]))
                             for r in ranks)
        curve[n] = {"mean_sd": statistics.mean(sds),
                    "p_modal": modal[1] / len(ranks), "mean_kendall": kd,
                    "win_rates": {m: sum(1 for r in ranks if r[0] == m) / len(ranks)
                                  for m in models}}
        print(f"{n:>4}{statistics.mean(sds):>21.4f}"
              f"{modal[1]/len(ranks):>18.0%}{kd:>20.1%}")
    report["curve"] = curve

    print(f"\n  P(each judge places #1), by n:")
    print(f"  {'judge':22}" + "".join(f"{'n='+str(n):>8}" for n in curve))
    for m in models:
        print(f"  {SHORT(m):22}" +
              "".join(f"{curve[n]['win_rates'][m]:>8.0%}" for n in curve))

    # --- contested passages: where voting hides ambiguity rather than fixing it
    print()
    print("=" * 74)
    print("CONTESTED PASSAGES  (modal answer wins by a thin margin)")
    print("=" * 74)
    print("  A 17-13 split is a coin flip. Majority vote reports it as a confident")
    print("  answer, so the instability moves out of the score and into the rubric.")
    print()
    contested = []
    for m in models:
        for i, s in enumerate(pool[m]):
            ok = [tuple(x) for x in s if x is not None]
            if not ok:
                continue
            c = collections.Counter(ok).most_common()
            margin = c[0][1] / len(ok)
            if margin < 0.75:
                contested.append({"model": m, "id": passages[i]["id"],
                                  "margin": margin, "gold": golds[i],
                                  "modal": list(c[0][0]),
                                  "split": {" ".join(k) or "(none)": v for k, v in c}})
    contested.sort(key=lambda d: d["margin"])
    by_passage = collections.Counter(d["id"] for d in contested)
    print(f"  {len(contested)} (judge, passage) cells below a 75% modal margin, "
          f"over {len(by_passage)} distinct passages")
    print(f"\n  passages contested by the most judges:")
    for pid, cnt in by_passage.most_common(6):
        gold = next(g for p, g in zip(passages, golds) if p["id"] == pid)
        print(f"    {pid}  contested by {cnt}/{len(models)} judges  "
              f"gold={gold or '(none)'}")
    print(f"\n  tightest splits:")
    for d in contested[:6]:
        print(f"    {SHORT(d['model']):20} {d['id']}  margin {d['margin']:.0%}  "
              f"gold={d['gold'] or '(none)'}  {d['split']}")
    report["contested"] = contested
    report["contested_by_passage"] = dict(by_passage)

    # --- bias vs variance: what is voting's ceiling? --------------------------
    print()
    print("=" * 74)
    print("BIAS vs VARIANCE  -- the ceiling self-consistency can reach")
    print("=" * 74)
    print("  Voting recovers the judge's MODAL answer. Where the mode is already")
    print("  wrong, more samples only raise confidence in the wrong label.")
    print()
    print(f"  {'judge':22}{'n=1 mean':>10}{'n=10':>8}{'ceiling':>9}{'gain':>8}"
          f"{'errors: bias':>14}{'variance':>10}")
    bias = {}
    for m in models:
        n1 = statistics.mean([r["scores"][m] for r in runs1])
        n10 = statistics.mean([r["scores"][m] for r in runs10])
        # ceiling = vote over the whole pool, the best estimate of the mode
        ceil_f, ceil_preds = score(golds, pool[m], args.beta)
        # split wrong passages into "mode is wrong" (bias) vs "mode is right but
        # a 10-sample vote could miss it" (variance)
        b = v = 0
        for i, s in enumerate(pool[m]):
            ok = [tuple(x) for x in s if x is not None]
            if not ok:
                continue
            modal_ok = tuple(ceil_preds[i]) == tuple(golds[i])
            if not modal_ok:
                b += 1
            elif collections.Counter(ok).most_common(1)[0][1] / len(ok) < 0.75:
                v += 1
        bias[m] = {"n1": n1, "n10": n10, "ceiling": ceil_f,
                   "bias_errors": b, "variance_errors": v}
        print(f"  {SHORT(m):22}{n1:>10.3f}{n10:>8.3f}{ceil_f:>9.3f}"
              f"{ceil_f-n1:>+8.3f}{b:>14}{v:>10}")
    report["bias_variance"] = bias
    tot_b = sum(d["bias_errors"] for d in bias.values())
    tot_v = sum(d["variance_errors"] for d in bias.values())
    print(f"\n  across all judges: {tot_b} passages the mode gets WRONG (voting "
          f"cannot help),")
    print(f"                     {tot_v} passages merely shaky (voting helps here)")
    print(f"  mean accuracy gain from n=1 -> unlimited voting: "
          f"{statistics.mean([d['ceiling']-d['n1'] for d in bias.values()]):+.3f}")

    # --- Q2: does voting buy significance? -----------------------------------
    print()
    print("=" * 74)
    print("Q2  PAIRED BOOTSTRAP AT n=10 -- is the ranking now SUPPORTED?")
    print("=" * 74)
    base = runs10[0]
    top = base["ranking"][0]
    print(f"  top judge (run1): {SHORT(top)}\n")
    print(f"  {'vs judge':22}{'delta':>9}{'95% CI':>20}{'p(d<=0)':>10}")
    cl = clusters_of(passages, args.cluster_key) if args.cluster_key != "none" else None
    print(f"  (resampling unit: {args.cluster_key}, "
          f"{len(set(cl)) if cl else len(passages)} clusters)\n")
    boot = {}
    for m in base["ranking"][1:]:
        bs = paired_bootstrap(golds, base["preds"][top], base["preds"][m],
                              args.beta, clusters=cl)
        boot[m] = bs
        d = base["scores"][top] - base["scores"][m]
        print(f"  {SHORT(m):22}{d:>+9.3f}"
              f"{'[%+.3f, %+.3f]' % (bs['lo'], bs['hi']):>20}"
              f"{bs['p_leq_0']:>10.4f}")
    report["n10_bootstrap_vs_top"] = boot
    sig = [m for m, b in boot.items() if b["p_leq_0"] < 0.05]
    print(f"\n  judges separated from #1 at p<0.05: "
          f"{', '.join(SHORT(m) for m in sig) if sig else 'NONE'}")
    report["n10_separated_from_top"] = sig

    # --- variance decomposition ----------------------------------------------
    print()
    print("=" * 74)
    print("WHERE THE UNCERTAINTY ACTUALLY LIVES")
    print("=" * 74)
    judge_sd = statistics.mean(
        [statistics.stdev([r["scores"][m] for r in runs1]) for m in models])
    vote_sd = curve.get(11, curve[max(curve)])["mean_sd"]
    ci_w = statistics.mean([b["hi"] - b["lo"] for b in boot.values()]) if boot else 0
    print(f"  judge sampling noise, n=1   (sd of macro-F)     {judge_sd:.4f}")
    print(f"  judge sampling noise, voted (sd of macro-F)     {vote_sd:.4f}")
    print(f"  eval-set noise, n=50 passages (mean CI width)   {ci_w:.4f}")
    print(f"\n  voting removed {(1-vote_sd/judge_sd)*100:.0f}% of judge sampling noise")
    print(f"  but the eval set is still {ci_w/max(vote_sd,1e-9):.0f}x wider "
          f"than the residual judge noise")
    report["variance"] = {"judge_sd_n1": judge_sd, "judge_sd_voted": vote_sd,
                          "eval_ci_width": ci_w}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
