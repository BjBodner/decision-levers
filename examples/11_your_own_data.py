"""11: your own data. Load a labelled CSV, split it, and compare the approaches on YOUR held-out items.

    python examples/11_your_own_data.py                                   # the bundled 60-row example CSV
    python examples/11_your_own_data.py --csv tickets.csv --text-col body --label-col queue \\
        --question "Which queue should handle this ticket?" --jev

The paper's advice at about 100 labels per class is to compare the local combination with the retrieval prompt on
your own held-out data, not to take its datasets' answer for yours. This script does that comparison end to end:

1. split the rows so no near-duplicate pair straddles pool and held-out items;
2. split the held-out items again: 'calib' fits the cascade threshold and the tempered exponents (cross-fitted),
   'test' is only scored;
3. report accuracy, macro-F1 and worst-class recall for each approach, and what each one sends to the vendor.

The bundled CSV is 60 hand-written HR requests (4 teams), far too small for evidence: replace it with yours.
"""
from pathlib import Path

from _common import backend_and_args

from decision_levers import (ClassifierFirstCascade, LabelledPool, Request, TunedLR, add_examples, combine,
                             fit_exponents, knn_vote, load_csv, tempered_combine)
from decision_levers.evaluate import class_metrics, holdout_split

HERE = Path(__file__).resolve().parent


def add_args(ap):
    ap.add_argument("--csv", default=str(HERE / "data" / "hr_requests.csv"), help="labelled CSV with a header row")
    ap.add_argument("--text-col", default="message")
    ap.add_argument("--label-col", default="team")
    ap.add_argument("--question", default="Which team should handle this employee request?")
    ap.add_argument("--test-frac", type=float, default=0.5, help="share of rows held out (calib + test)")
    ap.add_argument("--seed", type=int, default=0)


be, args = backend_and_args(__doc__, add_args)
rows = load_csv(args.csv, args.text_col, args.label_col)
texts, labels = [t for t, _ in rows], [lab for _, lab in rows]
options = list(dict.fromkeys(labels))                     # order of first appearance; keep it fixed
train_idx, held_idx, groups = holdout_split(texts, test_frac=args.test_frac, seed=args.seed)
calib_idx, test_idx = held_idx[: len(held_idx) // 2], held_idx[len(held_idx) // 2:]
pool = LabelledPool.from_pairs((texts[i], labels[i]) for i in train_idx)
prior = pool.shares()                                     # replace with deployment-period shares if you have them
print(f"{len(rows)} rows, {len(options)} labels; pool {len(pool)}, calib {len(calib_idx)}, test {len(test_idx)} "
      f"(near-duplicates kept on one side)")

lr = TunedLR(pool, options).fit()


def request(text):
    return Request(state=text, question=args.question, options=options)


def answers(idx):
    ts = [texts[i] for i in idx]
    return {"bare": be.decide_many([request(t) for t in ts]),
            "prompt": be.decide_many([add_examples(request(t), pool) for t in ts]),
            "vote": [knn_vote(t, pool, options, k=10, prior=prior) for t in ts],
            "lr": lr.predict(ts, prior=prior)}


cal, tst = answers(calib_idx), answers(test_idx)
cal_gold, gold = [labels[i] for i in calib_idx], [labels[i] for i in test_idx]

# tempered exponents fitted on calib (cross-validated), applied to test
fit = fit_exponents(cal["bare"], [cal["vote"], cal["lr"]], cal_gold, cv=3, groups=[groups[i] for i in calib_idx])
print(f"exponents fitted on calib (bare, vote, LR): {tuple(round(w, 2) for w in fit['weights'])}; "
      f"cross-fitted log-loss {fit['crossfit']['log_loss']:.3f} vs plain product {fit['plain']['log_loss']:.3f} "
      f"(keep the plain product unless the cross-fitted number is lower)")

# classifier-first cascade: threshold chosen on calib, with a cross-fitted estimate of what it really does
cas = ClassifierFirstCascade().fit(cal["lr"], cal["prompt"], cal_gold, max_acc_loss=0.005, cv=3,
                                   groups=[groups[i] for i in calib_idx])
cf = cas.report["crossfit"]
print(f"cascade on calib: in-sample (optimistic) {cas.report['share_sent']:.0%} sent, accuracy "
      f"{cas.report['accuracy']:.0%}; cross-fitted {cf['share_sent']:.0%} sent, accuracy {cf['accuracy']:.0%}")
send = cas.route(tst["lr"])
cascade = [p if s else lp for p, lp, s in zip(tst["prompt"], tst["lr"], send)]

arms = {
    "bare (sends the item)": tst["bare"],
    "kNN-20 prompt (item + pool texts)": tst["prompt"],
    "bare x vote x LR (item only)": [combine(b, v, p) for b, v, p in zip(tst["bare"], tst["vote"], tst["lr"])],
    "tempered, paper exponents": [tempered_combine(b, v, p) for b, v, p in zip(tst["bare"], tst["vote"], tst["lr"])],
    "tempered, your exponents": [combine(b, v, p, weights=fit["weights"])
                                 for b, v, p in zip(tst["bare"], tst["vote"], tst["lr"])],
    f"cascade ({send.mean():.0%} sent, with pool texts)": cascade,
}
print(f"\ntest items ({len(gold)}):")
for name, ans in arms.items():
    m = class_metrics(ans, gold)
    print(f"  {name:<40} accuracy {100 * m['accuracy']:5.1f}%  macro-F1 {100 * m['macro_f1']:5.1f}  "
          f"worst-class recall {100 * m['worst_class_recall']:5.1f} ({m['worst_class']})")
print("\nWith a few dozen items these differences are noise. Freeze the comparison and confirm it on enough held-out "
      "items before you choose (see examples/09).")
