# Algorithms

Pseudocode for each approach in this package and for the evaluation protocol. Each box lists inputs, output and
steps, and names the function that implements it. Notation: `x` is the text to decide on (the *state*), `O` the
ordered option list, `D` a labelled pool of `(text, label)` pairs, `M` the decision model, which returns a
probability vector over `O`. `sim(a, b)` is cosine similarity (TF-IDF on word 1-2-grams by default, or dense
sentence embeddings).

---

## A. Levers that add information

### Algorithm 1. Bare request (the reference)

```
Input:   x, instruction q, options O
Output:  p over O
1. request <- (state = x, question = q, criteria = {o: null for o in O})
2. p <- M(request)
```
`Request(x, q, O)`; `backend.decide(request)`.

### Algorithm 2. Retrieved examples in the option descriptions (kNN-20 prompt)

```
Input:   x, q, O, pool D, k = 20, per-label cap c (3; 10 when |O| = 2)
Output:  p over O
1. N <- the k rows of D with the highest sim(x, text), most similar first
2. S <- []; for (label, text) in N: if count(label in S) < c: append (label, text) to S
3. for each option o:
       if S has texts t1..tm labelled o: desc(o) <- 'Examples: "t1"; "t2"; ...; "tm"'   (replaces any description)
       else: desc(o) <- unchanged
4. p <- M(state = x, question = q, criteria = desc)
Note: pool texts are sent to the vendor.
```
`add_examples(request, pool, k=20)`; the selection alone is `select_examples`.

### Algorithm 3. Definitions, and neutral ids

```
Input:   request, definitions def(o) (one line each)
Output:  request'
1. for each option o: desc(o) <- def(o) if def(o) exists, else unchanged
Ids variant (to test or to hide names):
2. id(o) <- prefix + two-digit rank of o in alphabetical order        (option_01, option_02, ...)
3. send criteria {id(o): desc(o)} in the original option order; map answers back with id -> o
```
`add_definitions`, `rename_to_ids`, `map_back`.

### Algorithm 4. Class shares: stated, or applied after the call

```
Input:   request, true shares s(o) (from labelled data in the deployment period)
Stated:
1. sentence <- "Approximate share of past cases by option: " + join(", ", "o NN%" for o by decreasing s(o)) + "."
2. question <- question + " " + sentence;  p <- M(request)
Post hoc (no extra tokens), from the bare answer p:
3. p'(o) <- p(o) * s(o) / b(o), renormalised, where
       b(o) = 1                       (multiply by the shares), or
       b(o) = mean bare p(o) over unlabelled items, floored at 1e-4   (contextual calibration), or
       b(o) = share of o in a classifier's training pool               (prior shift)
```
`state_shares`, `correct_shares`, `mean_probs`.

### Algorithm 5. Packing several questions about one state

```
Input:   state x, main question Qm, other questions Q1..Qn about the same x
Output:  one answer per question
1. questions <- {Q1, ..., Qn, Qm}          (main question last)
2. send one request with state x and all questions; read each answer by name
```
`pack_questions(main, others)`; `backend.decide_packed(packed)`.

---

## B. Combining prior and evidence outside the prompt

### Algorithm 6. Similarity-weighted kNN vote with prior correction

```
Input:   x, pool D, options O, k = 10, eps = 1e-3, optional deployment prior pi(o)
Output:  v over O
1. v(o) <- eps for all o
2. for rank r, row i in the top-k rows of D by sim(x, text_i):
       v(label_i) += max(sim(x, text_i), 0) + 1e-6 / (r + 1)
3. v <- v / sum(v)
4. if pi given: v(o) <- v(o) * pi(o) / share_D(o), renormalised
```
`knn_vote`.

### Algorithm 7. Tuned, prior-corrected logistic regression

```
Input:   pool D (features from the retriever), options O, prior pi (optional)
Output:  classifier f(x) -> distribution over O
1. folds <- stratified K-fold, K = min(5, smallest class count)
2. for C in {0.1, 1, 10, 100, 1000}: out-of-fold logits; keep the C with the lowest log-loss
3. T <- the temperature in a log grid [0.02, 50] minimising out-of-fold NLL of softmax(logits / T)
4. refit on all of D with C;  f(x) = softmax(logits(x) / T)
5. if pi given: f(x)(o) <- f(x)(o) * pi(o) / share_D(o), renormalised
```
`TunedLR(pool, options).fit()`, `.predict(texts, prior)`.

### Algorithm 8. One bare call, combined locally (product of experts)

```
Input:   bare answer p (Algorithm 1), vote v (Algorithm 6), classifier f(x) (Algorithm 7), eps = 1e-3
Output:  e over O
1. e(o) <- (p(o) + eps) * (v(o) + eps) * (f(x)(o) + eps)
2. e <- e / sum(e)
Note: no pool text is sent to the vendor.
```
`combine(bare, vote, lr)`.

### Algorithm 9. Classifier-first cascade

```
Input:   held-out items with gold y_i, local answers l_i, model answers m_i (e.g. Algorithm 2);
         target: max accuracy loss delta (default 0.005), OR min recall rho for a positive class
Output:  threshold t
1. conf_i <- max_o l_i(o)
2. grid <- the 201 quantiles of conf, plus -1 and 1 + 1e-9
3. full <- accuracy of m on the held-out items
4. for t in grid:
       local_i <- conf_i > t;   pred_i <- argmax l_i if local_i else argmax m_i
       feasible if  accuracy(pred) >= full - delta      (accuracy target)
                or  recall_positive(pred) >= rho         (recall target)
5. t <- the feasible threshold with the smallest share sent (ties: higher accuracy); none feasible -> send all
Deploy: answer locally iff conf > t, else call the model.  Freeze t before testing.
```
`ClassifierFirstCascade().fit(local, model, gold, max_acc_loss=...)` or `fit(..., min_recall=..., positive=...)`.

### Algorithm 10. Labels-only shortlist

```
Input:   x, request with options O, pool D, k = 5
Output:  p over O, and whether a call was made
1. L <- distinct labels of the k nearest rows of D, in rank order, restricted to O
2. if |L| = 1: return p(L1) = 1, others 0;  no call
3. p <- M(request with options restricted to L, in the request's order); p(o) <- 0 for o not in L
```
`decide_with_shortlist(request, pool, backend, k=5)`.

---

## C. Evaluation protocol: screen -> freeze -> confirm

### Algorithm 11. Near-duplicate-aware holdout split

```
Input:   texts, threshold tau = 0.9, test fraction f, seed
Output:  train indices, test indices
1. group exact duplicates after lower-casing and stripping punctuation
2. link two groups if TF-IDF cosine >= tau; merge links transitively (union-find)
3. shuffle the groups (seeded); move whole groups to test until it holds >= f of the texts
4. the pool is drawn from train only
```
`near_duplicate_groups`, `holdout_split`.

### Algorithm 12. Null controls and the promotion floor

```
Input:   backend, held-out requests with gold, step = 0.01
Output:  null effects; floor
1. A <- answers to the requests; B <- answers to an identical resend (repeat index 1)
2. C <- answers after a content-free edit (a meaningless "[ref NNNN]" tag appended to the instruction)
3. for each null arm (B, C): mean change in clipped gold log-prob, accuracy change, flip rate against A
4. floor <- the smallest multiple of step strictly above the largest |null effect|
   (calibrate on null arms you did not tune on; the paper used 0.04, above 53 nulls)
```
`null_controls`, `promotion_floor`.

### Algorithm 13. Paired cluster bootstrap

```
Input:   per-item paired differences d_i (treatment - control), cluster ids g_i, B draws
Output:  estimate, interval, p-values
1. for b in 1..B: resample clusters with replacement; mean_b <- (sum of d in drawn clusters) / (items drawn)
2. interval <- quantiles of mean_b (90%: 5th and 95th)
3. p_sup = (1 + #{mean_b <= 0}) / (B + 1);  p_harm = (1 + #{mean_b >= 0}) / (B + 1)
   p_noninf = (1 + #{mean_b <= -margin}) / (B + 1)
   p_tost = max of the two one-sided p-values against -margin and +margin
Clusters: near-duplicate groups; the gold class when there are 50 or more options.
```
`bootstrap_means`, `paired_bootstrap`, `tost`, `p_value`.

### Algorithm 14. Screen, freeze, confirm

```
Input:   candidate levers, labelled data, a margin (e.g. 2 accuracy points) for "no worse" claims
Output:  confirmed decisions
Screen (development items):
1. split with Algorithm 11; compute nulls and the floor with Algorithm 12
2. for each lever: paired differences against its control; promote iff
       the 90% interval excludes 0  AND  effect >= floor  AND  flip rate > A/A flip rate
       (and, with several instruction paraphrases, the sign agrees across them)
Freeze:
3. write down each hypothesis, test type (superiority / non-inferiority / equivalence), margin, item set,
   exact request texts and thresholds; hash them (SHA-256) before any confirmatory call
Confirm (held-out items never used above):
4. run exactly the frozen arms; score paired accuracy with Algorithm 13
5. Holm step-down over the whole family at alpha = 0.05; report every test, including the failures
```
`compare` (screening rule), `freeze`, `holm`.
