"""Rank LLM judges on multi-label classification of annual-report passages.

Labels: adoption, risk, vendor, harm (subset per passage, possibly empty).
Gold labels come from data/passages.jsonl (hand-labeled per RUBRIC.md).
Outputs results/results.json with per-passage predictions per model, so the
site can recompute F_beta / bootstrap client-side.

Usage:
  ANTHROPIC_API_KEY=... OPENAI_API_KEY=... python judge_eval.py [--models m1,m2]
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
import urllib.request

LABELS = ["adoption", "risk", "vendor", "harm"]

INSTRUCTIONS = """You are labeling passages from corporate annual reports for an AI-disclosure observatory.

Assign every label that applies from this set (multi-label; the empty set is valid):

- adoption: the reporting company itself uses, deploys, develops, pilots, or integrates AI in its operations, products, or services. Not: industry commentary, pure aspiration with no commitment, or customers'/competitors' use.
- risk: the passage identifies AI as a source of POTENTIAL adverse outcome for the company (operational, model error, bias, security, regulatory, reputational, IP, workforce, competitive). Boilerplate risk-factor language counts. Not: AI used as a risk-mitigation tool (that is adoption).
- vendor: the company obtains AI capability from, or depends on, an external party (third-party models, cloud AI services, AI features in licensed software, AI-critical compute suppliers framed as a dependency). Not: generic IT outsourcing with no AI stated, or the company selling AI to others.
- harm: a negative AI-related event that HAS HAPPENED or is concretely in progress (incident, filed litigation or regulatory action, discovered bias failure, realized loss, layoffs attributed to AI). Potential-only language stays "risk".

The mere presence of the word "AI" never suffices for any label. Negated statements ("we do not use AI") get no label.

Respond with ONLY a JSON object: {"labels": [...]} using only strings from ["adoption","risk","vendor","harm"]. No prose."""


def load_env(path=".env"):
    """Populate os.environ from .env so the script runs without a shell wrapper."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def http_json(url, headers, payload, timeout=120):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# Reasoning models spend output budget on thinking before emitting any text. At
# the original 200 this truncated claude-opus-5 mid-thought on ~12% of samples:
# stop_reason=max_tokens, 200/200 thinking tokens, empty text -> unparseable ->
# silently scored as "predicted nothing". The OpenAI path caps nothing, so the
# tight cap penalised Anthropic judges only.
ANTHROPIC_MAX_TOKENS = 2000


def call_anthropic(model, passage):
    resp = http_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
        {
            "model": model,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "system": INSTRUCTIONS,
            "messages": [{"role": "user", "content": f"Passage:\n{passage}"}],
        },
    )
    text = "".join(b.get("text", "") for b in resp["content"])
    if resp.get("stop_reason") == "max_tokens" and not text.strip():
        # never let truncation masquerade as an empty-set prediction
        raise RuntimeError(f"truncated at max_tokens with no answer text ({model})")
    return text


def call_openai(model, passage):
    resp = http_json(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        {
            "model": model,
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": f"Passage:\n{passage}"},
            ],
        },
    )
    return resp["choices"][0]["message"]["content"]


def parse_labels(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        labels = json.loads(m.group(0)).get("labels", [])
        return sorted(set(l for l in labels if l in LABELS))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def judge(provider_model, passages, workers=8, samples=1, rng=None):
    """Draw `samples` independent judgements per passage.

    Returns preds[passage_idx][sample_idx] -> label list. Each sample is its own
    API call at the provider's default temperature, so the spread across samples
    IS the judge's stochastic noise. A sample that never parses is recorded as
    None (not []) so vote counts can exclude it instead of silently voting empty.
    """
    provider, model = provider_model.split(":", 1)
    call = {"anthropic": call_anthropic, "openai": call_openai}[provider]
    rng = rng or random.Random(0)

    def one(job):
        p, _ = job
        for attempt in range(5):
            try:
                labels = parse_labels(call(model, p["text"]))
                if labels is not None:
                    return labels
            except Exception as e:  # rate limits, transient network
                if attempt == 4:
                    print(f"  {model} FAILED on {p['id']}: {e}", file=sys.stderr)
                    return None
            # exponential backoff w/ jitter: at 6 models x 50 passages x k samples
            # the un-delayed retry loop in the n=1 version trips 429s constantly
            time.sleep(min(2 ** attempt, 16) * (0.5 + rng.random()))
        return None

    jobs = [(p, s) for p in passages for s in range(samples)]
    with cf.ThreadPoolExecutor(workers) as ex:
        flat = list(ex.map(one, jobs))
    return [flat[i * samples:(i + 1) * samples] for i in range(len(passages))]


def majority_vote(samples, threshold=0.5):
    """Per-label majority over a passage's samples; unparsed samples abstain.

    Multi-label, so each label is an independent binary vote: include it when it
    appears in strictly more than `threshold` of the parseable samples. At the
    0.5 default an exact tie (5-of-10) excludes the label -- ties are only
    reachable at even n, which is why odd n is the cleaner choice.
    """
    ok = [s for s in samples if s is not None]
    if not ok:
        return []
    counts = collections.Counter(l for s in ok for l in s)
    return sorted(l for l in LABELS if counts[l] > threshold * len(ok))


def prf(golds, preds, beta):
    """Per-class and micro/macro precision, recall, F_beta over the whole set.

    macro averages only over classes that actually have gold positives here. A
    class with no gold instances carries no evidence about judge quality, and
    the p=r=1 vacuous case scores it 1.0 -- free points. That mattered: `harm`
    has 2 gold passages in 50, so 13% of paired-bootstrap resamples contain no
    harm passage at all, which turned 25% of macro into the constant 1.0 and
    forced that component of the paired delta to exactly 0, narrowing every CI.

    The support set is a function of `golds` only, never of the predictions, so
    both judges in a paired comparison are always averaged over the same classes.
    """
    b2 = beta * beta
    per, tps = {}, {"tp": 0, "fp": 0, "fn": 0}
    for lab in LABELS:
        tp = sum(1 for g, p in zip(golds, preds) if lab in g and lab in p)
        fp = sum(1 for g, p in zip(golds, preds) if lab not in g and lab in p)
        fn = sum(1 for g, p in zip(golds, preds) if lab in g and lab not in p)
        tps["tp"] += tp; tps["fp"] += fp; tps["fn"] += fn
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        f = (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0
        per[lab] = {"p": prec, "r": rec, "f": f, "tp": tp, "fp": fp, "fn": fn,
                    "support": tp + fn}
    mp = tps["tp"] / (tps["tp"] + tps["fp"]) if tps["tp"] + tps["fp"] else 1.0
    mr = tps["tp"] / (tps["tp"] + tps["fn"]) if tps["tp"] + tps["fn"] else 1.0
    micro = (1 + b2) * mp * mr / (b2 * mp + mr) if mp + mr else 0.0
    support = [l for l in LABELS if per[l]["support"]]
    # No gold labels at all -> macro is undefined. micro already handles the
    # all-negative case correctly (1.0 iff the judge also predicted nothing).
    macro = sum(per[l]["f"] for l in support) / len(support) if support else micro
    return {"per_class": per, "micro_f": micro, "macro_f": macro,
            "micro_p": mp, "micro_r": mr, "macro_support": support}


def clusters_of(passages, key="company"):
    """Cluster id per passage. Resampling unit for the bootstrap -- see below."""
    return [p[key] for p in passages]


def paired_bootstrap(golds, preds_a, preds_b, beta, agg="macro", B=10000, seed=0,
                     clusters=None):
    """CI for delta = F(a) - F(b), resampling with pairing preserved.

    `clusters` gives a group id per passage (company by default). Whole clusters
    are drawn with replacement and every passage inside a drawn cluster comes
    with it, because passages from one filer are not independent draws: risk
    boilerplate is copied forward, disclosure style is constant, and judge errors
    repeat across a document. Treating them as independent understates the spread
    of a fresh sample and yields intervals that are too narrow.

    The unit also fixes the estimand. Resampling passages asks "how would this
    judge do on more passages from these same companies?"; resampling companies
    asks "...on companies I have not seen?" -- which is the question an
    observatory reporting corpus-wide prevalence is actually asking.

    clusters=None makes every passage its own cluster, which is exactly the old
    passage-level bootstrap (bit-identical: same rng draws, same indices).

    Resample sizes vary when clusters differ in size. That is correct, not a bug
    -- cluster-size heterogeneity is a real source of uncertainty and the method
    is meant to propagate it. Caveat: with few clusters (rule of thumb <30-40)
    the cluster bootstrap is anticonservative.
    """
    rng = random.Random(seed)
    n = len(golds)
    if clusters is None:
        groups = [[i] for i in range(n)]
    else:
        by = collections.OrderedDict()
        for i, c in enumerate(clusters):
            by.setdefault(c, []).append(i)
        groups = list(by.values())
    c_n = len(groups)
    deltas = []
    for _ in range(B):
        idx = [i for _ in range(c_n) for i in groups[rng.randrange(c_n)]]
        g = [golds[i] for i in idx]
        fa = prf(g, [preds_a[i] for i in idx], beta)[f"{agg}_f"]
        fb = prf(g, [preds_b[i] for i in idx], beta)[f"{agg}_f"]
        deltas.append(fa - fb)
    deltas.sort()
    return {"lo": deltas[int(0.025 * B)], "hi": deltas[int(0.975 * B)],
            "p_leq_0": sum(1 for d in deltas if d <= 0) / B,
            "n_clusters": c_n}


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=(
        "anthropic:claude-haiku-4-5,anthropic:claude-sonnet-5,"
        "openai:gpt-5-mini,openai:gpt-4o-mini"))
    ap.add_argument("--passages", default="data/passages.jsonl")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=1,
                    help="independent judgements per passage; >1 enables majority vote")
    ap.add_argument("--vote-threshold", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--samples-out", default="results/samples.jsonl",
                    help="raw per-sample labels, so re-analysis costs no API calls")
    ap.add_argument("--results-out", default="results/results.json",
                    help="scored output; point it elsewhere to avoid clobbering "
                         "the committed run")
    ap.add_argument("--cluster-key", default="company",
                    help="bootstrap resampling unit ('company', 'chunk_id', "
                         "or 'none' for the old passage-level bootstrap)")
    args = ap.parse_args()

    passages = [json.loads(l) for l in open(args.passages) if l.strip()]
    golds = [sorted(p["gold"]) for p in passages]
    out = {"labels": LABELS, "beta": args.beta, "samples": args.samples,
           "vote_threshold": args.vote_threshold,
           "passages": [{"id": p["id"], "text": p["text"], "gold": sorted(p["gold"])}
                        for p in passages],
           "models": {}}

    os.makedirs("results", exist_ok=True)
    raw = open(args.samples_out, "w") if args.samples_out else None

    for pm in args.models.split(","):
        pm = pm.strip()
        print(f"judging with {pm} (n={args.samples}) ...")
        drawn = judge(pm, passages, workers=args.workers, samples=args.samples)
        if raw:
            for p, samples in zip(passages, drawn):
                raw.write(json.dumps({"model": pm, "id": p["id"],
                                      "samples": samples}) + "\n")
            raw.flush()
        preds = [majority_vote(s, args.vote_threshold) for s in drawn]
        failed = sum(1 for s in drawn for x in s if x is None)
        if failed:
            print(f"  WARNING {failed} unparseable samples abstained", file=sys.stderr)
        stats = prf(golds, preds, args.beta)
        out["models"][pm] = {"preds": preds, "stats": stats}
        print(f"  macro-F{args.beta}={stats['macro_f']:.3f} "
              f"micro-F{args.beta}={stats['micro_f']:.3f} "
              f"(P={stats['micro_p']:.3f} R={stats['micro_r']:.3f})")

    # tie-break by name: an exact tie must not rank by dict insertion order
    ranked = sorted(out["models"],
                    key=lambda m: (-out["models"][m]["stats"]["macro_f"], m))
    out["ranking"] = ranked
    print("\nranking (macro-F):", " > ".join(ranked))
    top = ranked[0]
    cl = None if args.cluster_key == "none" else clusters_of(passages, args.cluster_key)
    out["cluster_key"] = args.cluster_key
    out["bootstrap_vs_top"] = {}
    for m in ranked[1:]:
        bs = paired_bootstrap(golds, out["models"][top]["preds"],
                              out["models"][m]["preds"], args.beta, clusters=cl)
        out["bootstrap_vs_top"][m] = bs
        print(f"  d(top - {m}) 95% CI [{bs['lo']:+.3f}, {bs['hi']:+.3f}] "
              f"p(d<=0)={bs['p_leq_0']:.4f}")

    if raw:
        raw.close()
        print(f"wrote {args.samples_out}")
    with open(args.results_out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.results_out}")


if __name__ == "__main__":
    main()
