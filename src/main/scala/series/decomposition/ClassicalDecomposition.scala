package series.decomposition

/**
 * Classical time series decomposition.
 *
 * Decomposes a series into trend, seasonal, and irregular components
 * using centered moving averages for trend estimation.
 *
 * Two models:
 *   Additive:       Y_t = T_t + S_t + I_t
 *   Multiplicative: Y_t = T_t * S_t * I_t
 *
 * Steps:
 *   1. Estimate trend via centered moving average of order s
 *   2. Compute detrended series (Y - T or Y/T)
 *   3. Average detrended values by season to get seasonal factors
 *   4. Normalize seasonal factors (sum/average to zero/one)
 *   5. Compute irregular = detrended - seasonal (or detrended / seasonal)
 *
 * @param period  seasonal period (e.g. 12 for monthly, 4 for quarterly)
 * @param additive if true uses additive model, else multiplicative
 */
class ClassicalDecomposition(period: Int, additive: Boolean = true) {
  require(period >= 2, "period must be at least 2")

  /**
   * Decomposes the series.
   *
   * @param y input series (length >= 2 * period)
   * @return DecompResult with trend, seasonal, irregular components
   */
  def decompose(y: Array[Double]): DecompResult = {
    val n = y.length
    require(n >= 2 * period, s"series length $n must be at least 2*period=${2*period}")

    // Step 1: Centered moving average trend
    val trend = centeredMA(y, period)

    // Step 2: Detrend
    val detrended = Array.tabulate(n) { t =>
      if (trend(t).isNaN) Double.NaN
      else if (additive) y(t) - trend(t)
      else y(t) / math.max(trend(t), 1e-10)
    }

    // Step 3: Seasonal factors — average by season index
    val seasonal = computeSeasonalFactors(detrended, period, additive, n)

    // Step 4: Irregular
    val irregular = Array.tabulate(n) { t =>
      val s = seasonal(t)
      val tr = trend(t)
      if (tr.isNaN || s.isNaN) Double.NaN
      else if (additive) y(t) - tr - s
      else y(t) / (tr * s)
    }

    DecompResult(
      original  = y,
      trend     = trend,
      seasonal  = seasonal,
      irregular = irregular,
      period    = period,
      additive  = additive
    )
  }

  /**
   * Centered moving average of order s.
   * For even s: 2x(s/2) MA (average of two adjacent s/2 MAs).
   * Result has NaN at first and last floor(s/2) positions.
   */
  private def centeredMA(y: Array[Double], s: Int): Array[Double] = {
    val n      = y.length
    val result = Array.fill(n)(Double.NaN)
    val half   = s / 2

    if (s % 2 == 1) {
      // Odd: simple centered MA
      for (t <- half until n - half)
        result(t) = (t - half to t + half).map(y).sum / s
    } else {
      // Even: 2x(s/2) MA
      for (t <- half until n - half) {
        val sum1 = (t - half until t + half).map(y).sum
        val sum2 = (t - half + 1 to t + half).map(y).sum
        result(t) = (sum1 + sum2) / (2.0 * s)
      }
    }
    result
  }

  private def computeSeasonalFactors(
    detrended: Array[Double],
    s:         Int,
    additive:  Boolean,
    n:         Int
  ): Array[Double] = {
    // Group by season index and average
    val groups = Array.fill(s)(scala.collection.mutable.ArrayBuffer[Double]())
    for (t <- detrended.indices)
      if (!detrended(t).isNaN)
        groups(t % s) += detrended(t)

    val rawFactors = groups.map { g =>
      if (g.isEmpty) 0.0 else g.sum / g.length
    }

    // Normalize: additive -> subtract mean; multiplicative -> divide by mean
    val normalized =
      if (additive) {
        val mean = rawFactors.sum / s
        rawFactors.map(_ - mean)
      } else {
        val mean = rawFactors.sum / s
        rawFactors.map(_ / math.max(mean, 1e-10))
      }

    Array.tabulate(n)(t => normalized(t % s))
  }
}

case class DecompResult(
  original:  Array[Double],
  trend:     Array[Double],
  seasonal:  Array[Double],
  irregular: Array[Double],
  period:    Int,
  additive:  Boolean
) {
  val n: Int = original.length

  /** Seasonal strength: 1 - Var(remainder) / Var(seasonal + remainder) */
  def seasonalStrength: Double = {
    val valid = irregular.indices.filter(i =>
      !irregular(i).isNaN && !seasonal(i).isNaN)
    if (valid.isEmpty) return Double.NaN
    val seasPlusIrr = valid.map(i => seasonal(i) + irregular(i))
    val varIrr      = variance(valid.map(i => irregular(i)).toArray)
    val varTotal    = variance(seasPlusIrr.toArray)
    if (varTotal < 1e-10) 0.0 else 1.0 - varIrr / varTotal
  }

  /** Trend strength: 1 - Var(remainder) / Var(trend + remainder) */
  def trendStrength: Double = {
    val valid = irregular.indices.filter(i =>
      !irregular(i).isNaN && !trend(i).isNaN)
    if (valid.isEmpty) return Double.NaN
    val trendPlusIrr = valid.map(i => trend(i) + irregular(i))
    val varIrr       = variance(valid.map(i => irregular(i)).toArray)
    val varTotal     = variance(trendPlusIrr.toArray)
    if (varTotal < 1e-10) 0.0 else 1.0 - varIrr / varTotal
  }

  private def variance(x: Array[Double]): Double = {
    val mu = x.sum / x.length
    x.map(v => (v - mu)*(v - mu)).sum / x.length
  }

  override def toString: String = {
    val model = if (additive) "Additive" else "Multiplicative"
    f"Classical Decomposition ($model, period=$period, n=$n)\n" +
    f"  Seasonal strength: $seasonalStrength%.4f\n" +
    f"  Trend strength:    $trendStrength%.4f\n"
  }
}
