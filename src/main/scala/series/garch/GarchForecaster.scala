package series.garch

/**
 * H-step ahead variance forecasts for a fitted GARCH model.
 *
 * For GARCH(p,q), the h-step ahead variance forecast converges
 * to the unconditional variance as h increases — mean reversion
 * in volatility is a key feature of stationary GARCH.
 *
 * Everything needed for forecasting is carried by the GarchFit, so the
 * forecaster takes only the fit. The common path is `fit.forecast(h)`.
 *
 * @param fit GarchFit with parameter estimates
 */
class GarchForecaster(fit: GarchFit) {

  /**
   * Produces h-step ahead conditional variance forecasts.
   *
   * Uses the recursion:
   *   E[sigma2_{T+h}] = omega + (alpha + beta) * E[sigma2_{T+h-1}]
   * for h > max(p,q), which simplifies to geometric convergence
   * toward the unconditional variance.
   *
   * @param h forecast horizon
   * @return GarchForecast with variance forecasts and volatility
   */
  def forecast(h: Int): GarchForecast = {
    require(h > 0, "Forecast horizon must be positive")

    val omega = fit.omega
    val alpha = fit.alpha
    val beta  = fit.beta

    // Initialize with last known sigma2 and eps values
    val lastSigma2 = fit.sigma2.takeRight(math.max(fit.p, 1))
    val lastEps2   = fit.residuals.takeRight(math.max(fit.q, 1))
                       .map(e => e * e)

    val forecastedVar = Array.fill(h)(0.0)

    // Extended arrays for recursion
    // For h > q: E[eps2_{T+j}] = E[sigma2_{T+j}] (since E[z_t^2] = 1)
    // For h > p: use forecasted variances
    val extSigma2 = lastSigma2.toBuffer
    val extEps2   = lastEps2.toBuffer

    for (i <- 0 until h) {
      var v = omega
      // ARCH terms — use actual eps2 for past, forecasted sigma2 for future
      for (j <- 0 until fit.q) {
        val idx = extEps2.length - 1 - j
        if (idx >= 0) v += alpha(j) * extEps2(idx)
      }
      // GARCH terms
      for (j <- 0 until fit.p) {
        val idx = extSigma2.length - 1 - j
        if (idx >= 0) v += beta(j) * extSigma2(idx)
      }
      val s2 = math.max(v, 1e-8)
      forecastedVar(i) = s2
      extSigma2 += s2
      extEps2   += s2  // E[eps2_{T+h}] = E[sigma2_{T+h}]
    }

    GarchForecast(
      varianceForecasts     = forecastedVar,
      volatilityForecasts   = forecastedVar.map(math.sqrt),
      unconditionalVariance = fit.unconditionalVariance,
      h                     = h
    )
  }
}

/**
 * H-step ahead GARCH variance forecast results.
 *
 * @param varianceForecasts     conditional variance forecasts
 * @param volatilityForecasts   conditional volatility (sqrt of variance)
 * @param unconditionalVariance long-run unconditional variance
 * @param h                     forecast horizon
 */
case class GarchForecast(
  varianceForecasts:     Array[Double],
  volatilityForecasts:   Array[Double],
  unconditionalVariance: Double,
  h:                     Int
) {
  override def toString: String = {
    val header = f"${"h"}%5s  ${"variance"}%12s  ${"volatility"}%12s"
    val rows = (0 until h).map { i =>
      f"${i+1}%5d  ${varianceForecasts(i)}%12.6f  ${volatilityForecasts(i)}%12.6f"
    }
    val footer = f"\n  Unconditional variance = $unconditionalVariance%.6f"
    (header +: rows).mkString("\n") + footer
  }
}
