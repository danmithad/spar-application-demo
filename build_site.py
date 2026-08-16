"""Inject passages (+ results if present) into site/template.html -> site/index.html.

Usage: python3 build_site.py [results.json]   (default results/results.json)
"""
import json, os, sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"

passages = [json.loads(l) for l in open("data/passages.jsonl")]
data = {
    "passages": [{k: p[k] for k in ("id", "company", "year", "text", "gold", "gold_note")}
                 for p in passages],
    "models": {},
}
if os.path.exists(RESULTS):
    res = json.load(open(RESULTS))
    assert [p["id"] for p in res["passages"]] == [p["id"] for p in passages]
    data["models"] = {m: {"preds": v["preds"]} for m, v in res["models"].items()}

tpl = open("site/template.html").read()
blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
out = tpl.replace("__DATA_JSON__", blob)
assert "__DATA_JSON__" not in out
open("site/index.html", "w").write(out)
print(f"site/index.html written: {len(out)/1024:.0f} KB, models: {list(data['models'])}")
