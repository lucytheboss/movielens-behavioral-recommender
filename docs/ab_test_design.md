# A/B Test Design: Popularity-Penalised MF

## Background

Phase 1 established that the MovieLens rating distribution has a severe long-tail structure:
the top 20% of items receive ~80% of all ratings. Phase 2 confirmed that standard Matrix
Factorization perpetuates this bias — catalog coverage is low and long-tail exposure rate
is well below the 0.80 baseline you'd expect from a neutral model.

The structural cause: popular items appear in more training pairs, so their latent factors
and bias terms (`b_i`) receive more gradient updates and are better optimised. At inference
time, high-confidence popular items dominate every ranked list.

## Hypothesis

> A popularity penalty applied at inference time will increase long-tail exposure rate
> without an unacceptable drop in NDCG@10.

The intervention is intentionally minimal: **no retraining**, no architecture change.
We subtract a scaled log-popularity term from each item's model score before ranking.

## Variants

| Variant | Score function |
|---------|---------------|
| **Control** | `score(u, i) = μ + b_u + b_i + Pᵤ · Qᵢᵀ` (vanilla MF) |
| **Treatment** | `score(u, i) = μ + b_u + b_i + Pᵤ · Qᵢᵀ − α · log(count(i) + 1)` |

`count(i)` = number of times item `i` appears in the training set.  
`α` is the penalty strength — the hyperparameter under test.

The `+ 1` inside the log prevents division issues for unseen items and keeps the penalty
well-behaved for items with very few ratings.

## Metrics

### Primary (optimise)
| Metric | Direction | Definition |
|--------|-----------|------------|
| Long-tail exposure rate | ↑ | Fraction of top-K recs pointing to tail items (bottom 80% by rating count) |

### Guardrail (must not regress past threshold)
| Metric | Max allowed drop | Definition |
|--------|-----------------|------------|
| NDCG@10 | −5% relative | Mean normalised discounted cumulative gain at rank 10 |
| Catalog coverage | no constraint | Fraction of items recommended to ≥1 user (expect improvement) |
| RMSE | no constraint | Not directly affected — penalty is inference-only |

The guardrail threshold of −5% on NDCG is intentionally lenient for an offline experiment.
In a live system you would tighten this based on business tolerance.

## Experiment Protocol (Offline)

This is a fully offline experiment run against the train/test split from notebooks 03–04.
No live traffic is involved.

### Step 1 — Establish control baseline

Run notebook 04 as-is. Record:
- `ndcg_control`, `lte_control`, `coverage_control`

These are already computed. Use them directly.

### Step 2 — Compute item popularity weights

```python
# count(i) over the training set
item_pop = (
    train.groupby('movieId')['rating']
    .count()
    .rename('count')
)
# log-popularity (add 1 to handle cold items)
item_log_pop = np.log1p(item_pop)
```

### Step 3 — Sweep α values

Test α ∈ {0.0, 0.1, 0.2, 0.5, 1.0, 2.0}. For each α:

1. For every sampled user, recompute item scores with the penalty applied.
2. Re-rank items by penalised score.
3. Compute NDCG@10, long-tail exposure rate, catalog coverage.

```python
def penalised_recommend(model, user_id, item_log_pop, alpha, top_k=10, exclude_seen=True):
    u = model.user2idx.get(user_id)
    if u is None:
        return []
    scores = model.mu + model.b_u[u] + model.b_i + model.P[u] @ model.Q.T

    # apply penalty
    for item_id, log_pop in item_log_pop.items():
        i = model.item2idx.get(item_id)
        if i is not None:
            scores[i] -= alpha * log_pop

    if exclude_seen and hasattr(model, '_seen'):
        for item_id in model._seen.get(user_id, set()):
            i = model.item2idx.get(item_id)
            if i is not None:
                scores[i] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(model.idx2item[i], float(scores[i])) for i in top_indices]
```

### Step 4 — Select α

Plot the NDCG@10 vs long-tail exposure trade-off curve across α values.
Choose the smallest α where long-tail exposure rate exceeds the control value by a meaningful
margin (target: +10 percentage points absolute) while NDCG@10 stays within the −5% guardrail.

### Step 5 — Report

Produce a results table:

| α | NDCG@10 | Δ NDCG | LT Exposure | Δ LT Exposure | Coverage | Selected? |
|---|---------|--------|-------------|---------------|----------|-----------|
| 0.0 (control) | … | — | … | — | … | |
| 0.1 | … | … | … | … | … | |
| … | … | … | … | … | … | |

## What This Test Cannot Tell You

- **Actual user satisfaction**: NDCG is a proxy. A live test would measure CTR, watch-time, or explicit ratings on recommended items.
- **Cold-start users**: sampled users all have training history. The penalty may behave differently for new users with few ratings.
- **Temporal effects**: the train/test split is random, not time-based. A chronological split would better simulate real deployment.

## Implementation Location

The experiment belongs in a new notebook: `notebooks/05_popularity_penalty.ipynb`.

It should import `MatrixFactorization` from `src/mf_model.py` and the metric functions
(`ndcg_at_k`, `long_tail_exposure`, `catalog_coverage`) from notebook 04 — or refactor them
into `src/metrics.py` if reuse across notebooks becomes unwieldy.
