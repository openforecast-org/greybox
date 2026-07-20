---
name: explain-greybox
description: Explain and interpret greybox outputs in plain language — ALM model summaries (coefficients, sigma, information criteria), confidence/prediction intervals, forecast accuracy measures, stepwise/CALM selection, RMCB/Nemenyi method comparisons, stick() STI decomposition, distributions, and association measures. Use when the user asks what a result means, how to read a summary or plot, which metric/distribution to pick, or why a model behaves a certain way — in either the R package or the Python port.
---

# Explaining greybox

This skill makes greybox results **understandable**. The goal is not to add
features but to translate the package's numerical output into clear, correct,
plain-language explanations grounded in what the code actually computes.

## How to explain (principles)

1. **Read the code, don't guess.** greybox has both an R original (root) and a
   Python port (`python/src/greybox/`). When a number's meaning is unclear,
   trace the function that produced it before explaining. The two
   implementations are designed for parity — if the user is in one language,
   answer in that language's idiom but the statistics are the same.
2. **State the definition, then the interpretation.** First what the quantity
   *is* (the formula / what it measures), then what *this particular value*
   tells the user (is it large, small, significant, dominant?).
3. **Lead with the takeaway.** Start with the one-sentence conclusion ("Method A
   is best and indistinguishable from D"), then justify it.
4. **Use the units and the convention.** Be explicit about scale conventions
   that bite people: `MPE`/`MAPE` are **fractions, not percentages** in this
   package; variance uses `df_residual = n - k`; `asymmetry` is centred at 0.
5. **Point to the canonical reference.** Each area has a wiki page (see below)
   and a literature reference; cite them rather than inventing notation.
6. **Prefer a plot when one exists.** Many results carry a `plot()` method
   (`ALM`, `stick`/`StickResult`, `rmcb`/`RMCBResult`, `aid`) — suggest it and
   say what to look for.
7. **Flag uncertainty honestly.** If an interval is wide, an IC difference is
   tiny, or a p-value is borderline, say the evidence is weak rather than
   overclaiming.

## Interpreting the main outputs

### ALM model summary (`summary(model)` / `model.summary()`)
- **Coefficients**: effect of each regressor on the (link-transformed)
  response. Report sign, magnitude, and the t-based significance / CI; remind
  the user that for non-`dnorm` distributions the effect is on the
  distribution's location, possibly on a transformed scale (e.g. log).
- **Error standard deviation (sigma)**: residual scale, computed with
  `df_residual = n - k`. Smaller is tighter fit, but compare across models only
  via information criteria, not sigma alone.
- **Information criteria (AIC / AICc / BIC / BICc)**: relative model quality;
  only differences matter. Rules of thumb: ΔIC < 2 ≈ indistinguishable, 2–6
  positive, 6–10 strong, >10 decisive. AICc for small samples, BIC/BICc penalise
  complexity more.
- **Degrees of freedom**: `n - k`; warn when it is small (estimates unstable).

### Intervals (`predict` / `PredictionResult.mean/.lower/.upper`)
- Distinguish a **confidence** interval (uncertainty of the mean) from a
  **prediction** interval (uncertainty of a new observation) — the latter is
  wider. Variance is `X @ vcov @ X'`. The interval reflects the chosen
  `distribution`, so it can be asymmetric (e.g. `dlnorm`).

### Forecast accuracy measures (`measures()` and the individual metrics)
- Group them: **scale-dependent** (ME, MAE, MSE, RMSE), **percentage** (MPE,
  MAPE — fractions here), **scaled** (MASE, RMSSE — vs naive, 1.0 = naive),
  **relative** (rMAE, rRMSE — vs a benchmark, <1 beats it). Pick scale-free
  measures to compare across series; never average MAPE across series with very
  different levels. See [hm()](https://github.com/openforecast-org/greybox/wiki/measures)
  half-moment measures for asymmetry/extremity of the error distribution.

### Selection: `stepwise()` and `CALM()`
- `stepwise()` greedily adds regressors by IC and partial correlation — explain
  *why* a variable entered/left, not just the final set. `CALM()` does **not**
  pick one model; it **weights** candidates by IC (Akaike weights) — explain the
  combined coefficients as weighted averages and that weights express model
  uncertainty.

### RMCB — comparing forecasting methods (`rmcb()`)
- The **default `distribution="tukey"` is the classical Nemenyi/MCB test**, not
  a regression: methods are ranked per series, each method's **mean rank** is the
  statistic, the **critical distance** is the Studentised-range value, and the
  overall p-value is the **Friedman test**. Non-default distributions
  (`dnorm`/`dlnorm`/other) are the *regression* variants whose intervals/p-value
  come from a fitted model. Explain overlap = no significant difference, and read
  the `"mcb"` plot (intervals, highlighted best method) and `"lines"` plot
  (groups). See [RMCB wiki](https://github.com/openforecast-org/greybox/wiki/RMCB).

### `stick()` — Seasonality / Trend / Irregular decomposition
- `strength` entries are **shares of total sum of squares** and **sum to 1**, so
  read them as proportions. A dominant `trend` means model the growth first; a
  large `irregular` share means the series is noisy / hard to predict. See
  [EDA wiki](https://github.com/openforecast-org/greybox/wiki/EDA).

### Distributions and association
- For `distribution=` choices, explain the shape/tail/support trade-off (e.g.
  `dlaplace` for heavier tails than `dnorm`, `dlnorm` for positive
  multiplicative data, count families for counts). For
  `pcor`/`mcor`/`association`/`determination`, distinguish partial vs multiple
  correlation and that `association()` auto-picks the right measure by variable
  type.

## Reference map

| Topic | Wiki page | Python module |
|---|---|---|
| ALM, summary, intervals | `ALM` | `alm.py`, `predict.py`, `methods/summary.py` |
| Selection / combination | `stepwise`, `CALM` | `selection.py` |
| Accuracy / half-moment / quantile | `measures` | `point_measures.py`, `hm.py`, `quantile_measures.py` |
| Method comparison | `RMCB` | `rmcb.py` |
| STI decomposition | `EDA` | `stick.py` |
| Distributions | `distributions` | `distributions/` |
| Association | `association` | `association.py` |
| Diagnostics / outliers | `diagnostics` | `diagnostics.py` |
| R vs Python parity / conventions | `R-Python-differences` | — |

When the explanation hinges on a numeric value the user pasted, recompute or
re-read the producing function before committing to an interpretation, and note
any R-vs-Python convention differences from the `R-Python-differences` page.
