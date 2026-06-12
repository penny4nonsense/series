# series

A from-scratch time series library for Scala, built on [Breeze](https://github.com/scalanlp/breeze) with no other dependencies. `series` provides classical and modern time series methods with consistent APIs, exact statistical foundations, and a test for every component.

[![Tests](https://img.shields.io/badge/tests-452%20passing-brightgreen)]()
[![Scala](https://img.shields.io/badge/scala-2.13%20%7C%203.3-red)]()

## Why another time series library?

The JVM ecosystem has no time series library with the breadth of Python's statsmodels or R's `forecast`/`vars`/`rugarch`. `series` aims to fill that gap: a single coherent library of time series methods, written from scratch with proper statistical foundations rather than wrapping existing tools. Every estimator is tested — including finite-difference gradient checks on the neural network backward passes.

The univariate models (ARIMA/SARIMA, ETS, GARCH) are the most mature and heavily exercised part of the library. The multivariate and structural tooling (VAR, SVAR, VECM) is newer and, while tested for correctness, has seen less real-world use — the `0.x` version line reflects that the API is not yet frozen.

## Installation

```scala
libraryDependencies += "io.github.penny4nonsense" %% "series" % "0.1.0"
```

Cross-built for Scala 2.13 and 3.3.

## What's included

| Area | Methods |
|------|---------|
| **Univariate** | ARIMA, SARIMA, ETS (full 15-model taxonomy with automatic selection), GARCH |
| **Multivariate** | VAR, SVAR (Cholesky, Blanchard-Quah, sign restrictions), VECM |
| **State space** | Kalman filter, diffuse Kalman filter, RTS smoother |
| **Filters** | Hodrick-Prescott, Baxter-King, Christiano-Fitzgerald, Butterworth |
| **Decomposition** | Classical, STL (robust), X-11 |
| **Interpolation** | Linear, cubic spline, PCHIP, Kalman-based |
| **Neural** | RNN, LSTM, GRU (stacked, multivariate, MC-dropout intervals) |
| **Transforms & tests** | Differencing (incl. fractional), ADF/PP/KPSS, ACF/PACF, cointegration |

---

## Quick start

### ARIMA

Fit an ARIMA(1,1,1), inspect the estimates, and forecast 10 steps with prediction intervals.

```scala
import series.arima._

val model     = ArimaModel(p = 1, d = 1, q = 1)
val estimator = new ArimaEstimator(model)
val fit       = estimator.fit(y)          // y: Array[Double]

println(fit)                              // estimates, std errors, AIC/BIC

// Forecast 10 steps — the fit remembers the data it was trained on
val fc = fit.forecast(h = 10)
println(fc)                               // mean, lower95, upper95 per step
```

By default `fit` runs CSS for starting values, then exact MLE via the Kalman filter. Pass `cssOnly = true` for the fast approximate fit.

To forecast against a *different* series than the one used for fitting (e.g. updated data with the same parameters), use the forecaster directly:

```scala
val fc = new ArimaForecaster(model, fit).forecast(newData, h = 10)
```

### Automatic ETS

Let the library choose among the 15 admissible ETS models by AICc.

```scala
import series.smoothing._

val result = new AutoEts(period = 12).fit(y)

println(result.best)          // e.g. ETS(A,Ad,A) with fitted parameters
println(result.ranking)       // all candidates ranked by criterion

val forecast = new EtsForecaster(result.best).forecast(h = 12, level = 0.95)
```

Pass `AutoEts(period = 12, criterion = AutoEts.BIC)` to select by BIC or AIC instead. Use `period = 1` for non-seasonal data (selects among the 6 non-seasonal models).

### GARCH volatility

Fit a GARCH(1,1) to a return series and forecast conditional volatility.

```scala
import series.garch._

val fit = GarchModel(p = 1, q = 1).fit(returns)   // returns: Array[Double]

println(f"persistence = ${fit.persistence}%.3f")  // alpha + beta
println(fit)                                      // full parameter table

val vol = fit.forecast(h = 20)
println(vol)   // variance and volatility forecasts, converging to unconditional
```

### Vector autoregression

Fit a VAR, select lag order, test Granger causality, and compute impulse responses.

```scala
import series.`var`._         // `var` needs backticks — it's a Scala keyword
import breeze.linalg.DenseMatrix

// Y: DenseMatrix[Double], rows = time, cols = series
val sel = VarLagSelection.select(Y, maxLag = 8)
println(sel)                                    // AIC / BIC / HQIC per lag

val fit = VarModel(p = sel.bicLag, k = Y.cols).fit(Y)
println(s"stable: ${fit.isStable}")

// Does series 0 Granger-cause series 1?
val gc = GrangerCausality.test(fit, cause = 0, effect = 1)
println(gc)

// Orthogonalized impulse responses + variance decomposition
val irf  = new ImpulseResponse(fit)
val resp = irf.orthogonalIRF(h = 20)
val fevd = irf.fevd(h = 20)

// Forecast
val fc = fit.forecast(h = 12)
```

### STL decomposition

Decompose a seasonal series into trend, seasonal, and remainder using robust LOESS.

```scala
import series.decomposition._

val stl = new StlDecomposition(period = 12).decompose(y)

println(f"seasonal strength = ${stl.seasonalStrength}%.3f")
println(f"trend strength    = ${stl.trendStrength}%.3f")

val trend     = stl.trend       // Array[Double]
val seasonal  = stl.seasonal
val remainder = stl.remainder
```

### Recurrent neural networks

Train a stacked LSTM on a multivariate series and forecast with MC-dropout prediction intervals.

```scala
import series.rnn._
import breeze.linalg.DenseMatrix

// Y: DenseMatrix[Double], rows = time, cols = series
val fit = RnnModel(RnnConfig.medium).fit(Y)   // LSTM, 2 layers, 64 hidden units
println(fit)                                  // architecture + training history

// Point forecast
val point = fit.forecast(Y, h = 12)

// Forecast with 95% prediction intervals via MC dropout
val pi = fit.forecastWithIntervals(Y, h = 12, level = 0.95, nSamples = 200)
println(pi)                                   // mean, lower, upper per step per series
```

`RnnConfig` defaults are deliberately conservative (single-layer, 32 hidden units) — safe minimums that fail visibly rather than overfit silently. For real problems use `RnnConfig.medium` or `RnnConfig.large`, or tune directly:

```scala
val cfg = RnnConfig(
  cellType  = GRU,
  hiddenDim = 128,
  nLayers   = 2,
  window    = 24,        // lookback; -1 auto-computes from period/horizon
  horizon   = 12,
  dropout   = 0.2,
  lossType  = Huber,     // MSE, MAE, or Huber
  epochs    = 300
)
val fit = RnnModel(cfg).fit(Y)
```

All inputs are normalized internally; forecasts are returned on the original scale.

---

## Design principles

- **No dependencies beyond Breeze.** Everything — optimization, linear algebra glue, statistical distributions — is built on Breeze primitives.
- **Exact foundations.** ARIMA/SARIMA estimation uses the Kalman filter for exact MLE with diffuse initialization (Durbin-Koopman). VAR is exact OLS. VECM uses Johansen reduced-rank regression. The RNN gradients are verified against finite differences.
- **Consistent APIs.** Models follow a `Model → fit → Fit` shape, and result types carry `toString` methods that print readable summaries. Most fitted models — including ARIMA — forecast directly via `fit.forecast(h)`. (SARIMA currently uses an explicit `new SarimaForecaster(model, fit).forecast(y, h)`; unifying it is planned for a later release. ARIMA also exposes its explicit forecaster for forecasting against new data.)
- **Tested.** 452 tests across both Scala versions, covering dimensional correctness, known analytical results, statistical recovery on simulated data, and gradient correctness.

## A note on the neural module

The RNN forecaster recomputes layer activations during the backward pass rather than caching them, trading compute for memory simplicity. This is fine for typical time series lengths but is a known place to optimize for very long sequences. MC-dropout prediction intervals are a Bayesian approximation (Gal & Ghahramani 2016) and tend to be optimistic at long horizons — they are not calibrated coverage guarantees.

## License

MIT
