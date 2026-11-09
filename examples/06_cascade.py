"""06: classifier-first cascade. A local classifier answers the items it is sure about; the rest go to the model.

The threshold is fitted on HELD-OUT items (here the 'calib' split), then frozen and applied to 'test'. In the paper
a threshold frozen in advance kept moderation accuracy at 8.2% of items sent (confirmatory, accuracy only), but
the same threshold cost 7.3 points of precision on the rare toxic class (exploratory): pick a recall target for
rare positives, as shown at the end.
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
cas = ClassifierFirstCascade().fit(local(ct), [model(s) for s in ct], [t.label for t in calib], max_acc_loss=0.005)
print(f"fitted on {len(calib)} held-out items: threshold {cas.threshold:.3f}, report {cas.report}")

test = toy.load("test")
gold = [t.label for t in test]
answers, sent = cas.predict([t.text for t in test], local, model)
show(f"cascade ({sent:.0%} sent to the model)", answers, gold)
show("model on every item", [model(t.text) for t in test], gold)
show("local classifier only", local([t.text for t in test]), gold)

rare = ClassifierFirstCascade().fit(local(ct), [model(s) for s in ct], [t.label for t in calib],
                                    min_recall=0.5, positive="Bug report")
if rare.report["feasible"]:
    print(f"recall target (>= 50%) for the rare class 'Bug report': threshold {rare.threshold:.3f}, "
          f"share sent on calib {rare.report['share_sent']:.0%}")
else:
    print("recall target (>= 50%) for the rare class 'Bug report': no threshold reaches it on calib, "
          "so every item goes to the model")
