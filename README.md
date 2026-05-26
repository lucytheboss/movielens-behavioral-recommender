# MovieLens Behavioral Recommender

> **Can we diagnose *and* quantify popularity bias in a recommender system — and design an experiment to fix it?**

This project builds from diagnosis to intervention across two phases, using the MovieLens 25M dataset.

---

## Project Narrative

| Phase | Question | Approach |
|-------|----------|----------|
| **Phase 1** | Is long-tail under-exposure driven by power users, or is it structural? | Counterfactual sensitivity analysis |
| **Phase 2** | Does a real MF model perpetuate that bias — and how would we fix it? | MF from scratch + metrics + A/B test design |

---

## Phase 1 — Popularity Bias Diagnosis

**Finding**: Removing the top 1–5% most active users barely changes head/tail exposure ratios.
Popularity bias is a structural property of interaction data, not a user behaviour problem.

- Dataset: MovieLens 25M (~25M ratings, 162K users, 59K movies)
- Method: Counterfactual user stratification (Quiet / Active / Power / Ultra / Extreme)
- Key result: Top 20% of items receive **98.26%** of all ratings — and this holds regardless of which users are removed

---

## Phase 2 — Model Implementation & Experiment Design

### Matrix Factorization (from scratch)

Implemented collaborative filtering via MF with SGD — numpy only, no surprise/sklearn for the core algorithm.

$$\hat{r}_{ui} = \mu + b_u + b_i + p_u \cdot q_i$$

- User/item bias terms + L2 regularization + early stopping
- Best val RMSE: **0.9062** (epoch 33, patience=5)
- Hyperparameters: n_factors=20, lr=0.005, λ=0.05

### Evaluation Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| RMSE (test) | 0.9309 | Prediction accuracy |
| NDCG@10 | 0.9502 | Closed-world ranking quality (200 users) |
| Catalog Coverage | 0.07% (12/16,825) | Open-world — popularity bias confirmed |
| Long-tail Exposure | 0.5980 | vs ideal ~0.80 |

**Key finding**: High closed-world NDCG (0.95) but near-zero catalog coverage (0.07%) reveals the core tension — the model ranks known preferences well but defaults to popular items for open-ended recommendations, amplifying Phase 1's structural bias.

### A/B Test Design

Designed a full experiment to test a popularity-penalised variant:
- **Treatment**: subtract α·log(count(i)+1) from item scores at inference (no retraining needed)
- **Primary metrics**: Long-tail Exposure Rate ↑, NDCG@10 ≥ control
- **Guardrail metrics**: Session Length, CTR ≥ control
- Includes sample size calculation, novelty effect handling (1-week burn-in), and network effect discussion

See [`docs/ab_test_design.md`](docs/ab_test_design.md) for full design.

---

## Repo Structure

```
notebooks/
  01_load_and_validate.ipynb              # data loading and preprocessing
  02_user_behavior.ipynb          # Phase 1: counterfactual analysis
  03_matrix_factorization.ipynb   # Phase 2: MF implementation and training
  04_evaluation_metrics.ipynb     # Phase 2: RMSE, NDCG, coverage, LTE
src/
  mf_model.py                     # MF class — numpy only, from scratch
docs/
  ab_test_design.md               # experiment design document
images/
  loss_curve.png                  # training curve
```

---

## Stack

Python · NumPy · Pandas · Matplotlib · Jupyter

---

## Key References

- Koren et al. (2009) "Matrix Factorization Techniques for Recommender Systems"
- Kohavi et al. "Trustworthy Online Controlled Experiments"
- Nguyen et al. (2014) "Exploring the filter bubble"
