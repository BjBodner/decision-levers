"""06: classifier-first cascade. A local classifier answers the items it is sure about; the rest go to the model.

The threshold is fitted on HELD-OUT items (here the 'calib' split), then frozen and applied to 'test'. In the paper
a threshold frozen in advance sent 8.2% of moderation items to the model and was no more than 1.5 points below the
prompt on accuracy (confirmatory, accuracy only), but it cost 7.3 points of precision on the rare toxic class
(exploratory): pick a recall or precision target for rare positives, as shown at the end.

A threshold chosen on the same items it is scored on is optimistic; cv=k reports a cross-fitted estimate.
Forwarded items go to the kNN-20 prompt, so they are sent WITH their nearest labelled pool texts.
"""
from _common import backend_from_args, show

from decision_levers import ClassifierFirstCascade, LabelledPool, TunedLR, add_examples, toy

be = backend_from_args(__doc__)
pool = LabelledPool.from_pairs((t.text, t.label) for t in toy.load("train"))
lr = TunedLR(pool, toy.OPTIONS).fit()
prior = toy.train_shares()


def local(texts):
    return lr.predict(texts, prior=prior)


def model(text):
    return be.decide(add_examples(toy.bare_request(text), pool))


calib = toy.load("calib")
ct = [t.text for t in calib]
calib_model = [model(s) for s in ct]
cas = ClassifierFirstCascade().fit(local(ct), calib_model, [t.label for t in calib], max_acc_loss=0.005, cv=4)
cf = cas.report["crossfit"]
print(f"fitted on {len(calib)} held-out items: threshold {cas.threshold:.3f}")
print(f"  in sample (optimistic): {cas.report['share_sent']:.0%} sent, accuracy {cas.report['accuracy']:.0%}")
print(f"  cross-fitted (4 folds): {cf['share_sent']:.0%} sent, accuracy {cf['accuracy']:.0%}, "
      f"target met out of fold: {cf['meets_target']}")

test = toy.load("test")
gold = [t.label for t in test]
answers, sent = cas.predict([t.text for t in test], local, model)
show(f"cascade ({sent:.0%} sent to the model)", answers, gold)
show("model on every item", [model(t.text) for t in test], gold)
show("local classifier only", local([t.text for t in test]), gold)

for name, target in (("recall >= 50%", {"min_recall": 0.5}),
                     ("precision >= 80%", {"min_precision": 0.8}),
                     ("recall >= 50% and precision >= 80%", {"min_recall": 0.5, "min_precision": 0.8})):
    rare = ClassifierFirstCascade().fit(local(ct), calib_model, [t.label for t in calib], positive="Bug report",
                                        **target)
    if rare.report["feasible"]:
        print(f"{name} for the rare class 'Bug report': threshold {rare.threshold:.3f}, share sent on calib "
              f"{rare.report['share_sent']:.0%} (recall {rare.report['recall']:.0%}, "
              f"precision {rare.report['precision']:.0%}; in sample)")
    else:
        print(f"{name} for the rare class 'Bug report': no threshold reaches it on calib, so every item goes "
              f"to the model")
