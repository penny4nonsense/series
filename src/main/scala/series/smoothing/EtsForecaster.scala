package series.smoothing

import breeze.linalg._

/**
 * H-step ahead forecasts for a fitted ETS model.
 *
 * Point forecasts are computed by iterating the state space recursion
 * forward. Prediction intervals use the analytical variance formulas
 * from Hyndman et al. (2008), Table 2.
 */
class EtsForecaster(fit: EtsFit) {

  /**
   * Produces h-step ahead forecasts with prediction intervals.
   *
   * @param h     forecast horizon
   * @param level confidence level for intervals (e.g. 0.95)
   * @return EtsForecast
   */
  def forecast(h: Int, level: Double = 0.95): EtsForecast = {
    require(h > 0, "horizon must be positive")
    require(level > 0 && level < 1, "level must be in (0,1)")

    val spec   = fit.spec
    val params = fit.params
    val ss     = new EtsStateSpace(spec, params)

    // Start from final filtered state
    val lastState = fit.states(fit.n).copy

    val means   = new Array[Double](h)
    val variances = new Array[Double](h)

    // Point forecasts: iterate state forward without noise
    var state = lastState.copy
    for (i <- 0 until h) {
      means(i) = ss.w(state)
      state    = ss.transition(state, 0.0)
    }

    // Forecast variances from cumulative c_j coefficients
    val cj = cCoefficients(h)
    for (i <- 0 until h) {
      variances(i) = params.sigma2 * (0 to i).map(j => cj(j) * cj(j)).sum
    }

    val z     = quantileNormal((1 + level) / 2)
    val lower = Array.tabulate(h)(i => means(i) - z * math.sqrt(math.max(variances(i), 0)))
    val upper = Array.tabulate(h)(i => means(i) + z * math.sqrt(math.max(variances(i), 0)))

    EtsForecast(means = means, lower = lower, upper = upper,
      variances = variances, h = h, level = level, spec = spec)
  }

  /**
   * Computes c_j coefficients for forecast variance.
   * c_0 = 1, c_j = phi^j * alpha + sum of seasonal terms.
   * Approximation: use alpha + (beta if trend) + (gamma if seasonal).
   */
  private def cCoefficients(h: Int): Array[Double] = {
    val p      = fit.params
    val spec   = fit.spec
    val c      = new Array[Double](h)
    c(0) = 1.0
    for (j <- 1 until h) {
      c(j) = p.phi * c(j-1) * (if (spec.hasTrend) 1.0 + p.beta else 1.0)
      if (spec.hasSeasonal && j % spec.period == 0) c(j) += p.gamma
    }
    c
  }

  private def quantileNormal(p: Double): Double = {
    // Rational approximation for normal quantile (Abramowitz & Stegun)
    val t = math.sqrt(-2 * math.log(1 - p))
    val c = Array(2.515517, 0.802853, 0.010328)
    val d = Array(1.432788, 0.189269, 0.001308)
    t - (c(0) + c(1)*t + c(2)*t*t) / (1 + d(0)*t + d(1)*t*t + d(2)*t*t*t)
  }
}

case class EtsForecast(
  means:     Array[Double],
  lower:     Array[Double],
  upper:     Array[Double],
  variances: Array[Double],
  h:         Int,
  level:     Double,
  spec:      EtsSpec
) {
  override def toString: String = {
    val sb = new StringBuilder
    sb.append(s"${spec.name} Forecasts (${(level*100).toInt}% intervals):\n")
    sb.append(f"  ${"h"}%5s  ${"mean"}%10s  ${"lower"}%10s  ${"upper"}%10s\n")
    for (i <- 0 until h)
      sb.append(f"  ${i+1}%5d  ${means(i)}%10.4f  ${lower(i)}%10.4f  ${upper(i)}%10.4f\n")
    sb.toString
  }
}
