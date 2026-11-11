"""12: replay the paper's flagship worked example: one Banking77 question, three real requests to Jev 1.13.

    python examples/12_replay_flagship.py           # offline: payload diffs and the recorded probabilities
    python examples/12_replay_flagship.py --jev     # also re-sends the 3 bodies (needs TYPESAFE_API_KEY;
                                                    # about 3,200 input tokens, roughly $0.0001 at list price)

The file ``real_payloads/flagship_banking77.json`` holds the exact wire bodies the paper sent and the answers Jev
returned (Figure 3 of the paper): bare, with three retrieved correctly labelled examples, and with the same three
texts under wrong labels. Offline, the script prints what changed between the requests, the recorded answers, and
checks that this package's ``Request`` rebuilds each body byte for byte. ``--jev`` re-sends them through a pinned
``JevBackend``; expect small differences, since identical resends kept the argmax for 97.3 to 99.6% of requests.

Banking77: Casanueva et al. 2020 (PolyAI), CC BY 4.0. See real_payloads/README.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a source checkout

from decision_levers import JevBackend, ReplayBackend, Request  # noqa: E402
from decision_levers.backends import KEY_ENVS, request_key  # noqa: E402

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
g = ap.add_mutually_exclusive_group()
g.add_argument("--mock", action="store_true", help="offline replay of the recorded answers (default)")
g.add_argument("--jev", action="store_true", help="also re-send the three bodies to the real API (paid)")
ap.add_argument("--file", default=str(HERE / "real_payloads" / "flagship_banking77.json"))
args = ap.parse_args()

doc = json.loads(Path(args.file).read_text(encoding="utf-8"))
reqs, gold = doc["requests"], doc["gold"]


def recorded(side):
    """The recorded probabilities without the file's '(other N options, summed)' row."""
    return {o: float(p) for o, p in side["probabilities"].items() if not o.startswith("(other ")}


def top(p, n=5):
    return "  ".join(f"{'*' if o == gold else ''}{o}={v:.2f}" for o, v in sorted(p.items(), key=lambda kv: -kv[1])[:n])


def short(v, width=150):
    s = "null" if v is None else repr(v)
    return s if len(s) <= width else s[: width - 3] + "..."


def diff(a, b):
    qa, qb = a["questions"]["q"], b["questions"]["q"]
    same = [k for k, x, y in (("state", a["state"], b["state"]), ("instructions", qa["instructions"],
                                                                     qb["instructions"])) if x == y]
    order = "same" if list(qa["criteria"]) == list(qb["criteria"]) else "DIFFERENT"
    print(f"    unchanged: {', '.join(same) or 'nothing'}; option names and order: {order} "
          f"({len(qb['criteria'])} options)")
    for o in qb["criteria"]:
        if qa["criteria"].get(o) != qb["criteria"][o]:
            print(f"    ~ {o}: {short(qa['criteria'].get(o))}\n    {'':>{len(o) + 4}}-> {short(qb['criteria'][o])}")


if args.jev:
    print(f"backend: TypeSafe Jev, model {doc['model']} (paid; cached on disk) plus the recorded answers")
else:
    print("backend: offline mock (replaying the recorded answers in the file; nothing is sent)")
print(f"{doc['dataset']['source']} {doc['dataset']['split']} item {doc['dataset']['item_id']} "
      f"({doc['dataset']['licence']}): {doc['state']!r}  gold = {gold}  model = {doc['model']}")
print(f"question: {reqs['1_bare']['wire_request']['questions']['q']['instructions']!r}\n")

names = list(reqs)
for i, name in enumerate(names):
    side = reqs[name]
    print(f"[{name}] {side['label']}  (billed input tokens {side['billed_input_tokens']:,})")
    if i:
        print(f"  what changed from [{names[i - 1]}]:")
        diff(reqs[names[i - 1]]["wire_request"], side["wire_request"])
    print(f"  recorded answer (top 5, * = gold): {top(recorded(side))}")
    print(f"  -> {side['argmax']} ({'right' if side['correct'] else 'wrong'}; p(gold) = {side['p_gold']:.2f})\n")

# This package builds the same bodies: Request(state, question, criteria).to_wire(model) is byte-identical.
rebuilt = [Request(state=s["wire_request"]["state"], question=s["wire_request"]["questions"]["q"]["instructions"],
                   options=s["wire_request"]["questions"]["q"]["criteria"]).to_wire(doc["model"])
           for s in reqs.values()]
same = all(request_key(doc["model"], r) == request_key(doc["model"], s["wire_request"])
           for r, s in zip(rebuilt, reqs.values()))
print(f"decision_levers.Request rebuilds all three bodies byte for byte: {same}")

replay = ReplayBackend([(s["wire_request"], recorded(s)) for s in reqs.values()])
agree = sum(max(p := replay.decide_wire(r)["q"], key=p.get) == s["argmax"] for r, s in zip(rebuilt, reqs.values()))
print(f"replayed offline through ReplayBackend: argmax agreement with the recorded answers {agree} of {len(reqs)}")

if args.jev:
    if not any(os.environ.get(k) for k in KEY_ENVS):
        print(f"\n--jev: no API key set ({' or '.join(KEY_ENVS)}), so nothing was sent.")
    else:
        be = JevBackend(doc["model"])
        print("\nre-sent to the API now (top 5, * = gold):")
        agree = 0
        for name, s in reqs.items():
            now = be.decide_wire(s["wire_request"])["q"]
            agree += max(now, key=now.get) == s["argmax"]
            print(f"  [{name}] {top(now)}")
        print(f"argmax agreement with the recorded answers: {agree} of {len(reqs)} "
              f"(identical resends agreed on 97.3-99.6% of requests in the paper)")
