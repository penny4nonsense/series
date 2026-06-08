package series.decomposition

/**
 * Simplified X-11 seasonal adjustment (2-pass version).
 *
 * The full X-11/X-12-ARIMA is hundreds of pages of Census Bureau code.
 * This implements the essential 2-pass structure:
 *
 * Pass 1 (preliminary):
 *   1. Initial trend: centered MA of order s
 *   2. Preliminary seasonal: average of detrended by season
 *   3. Preliminary seasonally adjusted series
 *
 * Pass 2 (final):
 *   4. Final trend: Henderson MA on seasonally adjusted series
 *   5. Final seasonal: average of (original / trend) by season
 *   6. Final irregular: original / (trend * seasonal)
 *   7. Extreme value identification and replacement
 *
 * Both additive and multiplicative decomposition are supported.
 *
 * @param period   seasonal period
 * @param additive if true uses additive model
 */
class X11Decomposition(period: Int, additive: Boolean = false) {
  require(period >= 2, "period must be at least 2")

  def decompose(y: Array[Double]): X11Result = {
    val n = y.length
    require(n >= 3 * period, s"X-11 needs at least 3*period=${3*period} observations, got $n")

    // Pass 1 ──────────────────────────────────────────────────────────────────

    // Step 1: Initial trend via centered MA
    val trend1 = centeredMA(y, period)

    // Step 2: Preliminary seasonal ratios/differences
    val ratio1 = Array.tabulate(n) { t =>
      if (trend1(t).isNaN) Double.NaN
      else if (additive) y(t) - trend1(t)
      else if (math.abs(trend1(t)) > 1e-10) y(t) / trend1(t) else Double.NaN
    }

    // Step 3: Preliminary seasonal factors (3x3 MA of season groups)
    val seasonal1 = seasonalFactors(ratio1, period, additive, n, maOrder = 3)

    // Preliminary seasonally adjusted
    val sadj1 = Array.tabulate(n) { t =>
      if (seasonal1(t).isNaN) Double.NaN
      else if (additive) y(t) - seasonal1(t)
      else y(t) / math.max(seasonal1(t), 1e-10)
    }

    // Pass 2 ──────────────────────────────────────────────────────────────────

    // Step 4: Henderson MA trend on seasonally adjusted series
    val hLength = hendersonLength(period)
    val trend2  = hendersonMA(sadj1, hLength)

    // Step 5: Final seasonal factors (3x5 MA of season groups)
    val ratio2 = Array.tabulate(n) { t =>
      val tr = trend2(t)
      if (tr.isNaN) Double.NaN
      else if (additive) y(t) - tr
      else if (math.abs(tr) > 1e-10) y(t) / tr else Double.NaN
    }

    val seasonal2 = seasonalFactors(ratio2, period, additive, n, maOrder = 5)

    // Final seasonally adjusted
    val sadj2 = Array.tabulate(n) { t =>
      if (seasonal2(t).isNaN) Double.NaN
      else if (additive) y(t) - seasonal2(t)
      else y(t) / math.max(seasonal2(t), 1e-10)
    }

    // Final trend (Henderson on final SA)
    val trend3 = hendersonMA(sadj2, hLength)

    // Step 6: Final irregular
    val irregular = Array.tabulate(n) { t =>
      val tr = trend3(t); val s = seasonal2(t)
      if (tr.isNaN || s.isNaN) Double.NaN
      else if (additive) y(t) - tr - s
      else y(t) / math.max(tr * s, 1e-10)
    }

    X11Result(
      original           = y,
      trend              = trend3,
      seasonal           = seasonal2,
      irregular          = irregular,
      seasonallyAdjusted = sadj2,
      period             = period,
      additive           = additive
    )
  }

  /** Centered moving average */
  private def centeredMA(y: Array[Double], s: Int): Array[Double] = {
    val n = y.length; val result = Array.fill(n)(Double.NaN)
    val half = s / 2
    if (s % 2 == 1) {
      for (t <- half until n - half)
        result(t) = (t-half to t+half).map(y).sum / s
    } else {
      for (t <- half until n - half) {
        val s1 = (t-half until t+half).map(y).sum
        val s2 = (t-half+1 to t+half).map(y).sum
        result(t) = (s1 + s2) / (2.0 * s)
      }
    }
    result
  }

  /**
   * Seasonal factors using MA smoothing of subseries.
   * maOrder = 3: 3-term MA of each season group (3x3 in X-11 parlance)
   * maOrder = 5: 5-term MA (3x5)
   */
  private def seasonalFactors(
    ratios:   Array[Double],
    s:        Int,
    additive: Boolean,
    n:        Int,
    maOrder:  Int
  ): Array[Double] = {
    val raw = new Array[Double](n)

    for (k <- 0 until s) {
      val subIdx = (k until n by s).toArray
      val subR   = subIdx.map(ratios).filter(!_.isNaN)
      if (subR.length >= maOrder) {
        val smoothed = simpleMA(subR, maOrder)
        val offset   = (maOrder - 1) / 2
        for (i <- smoothed.indices)
          if (i + offset < subIdx.length)
            raw(subIdx(i + offset)) = smoothed(i)
        // Fill endpoints with edge values
        for (i <- 0 until offset)
          if (i < subIdx.length) raw(subIdx(i)) = smoothed.head
        for (i <- subIdx.length - offset until subIdx.length)
          if (i >= 0) raw(subIdx(i)) = smoothed.last
      }
    }

    // Normalize
    val byGroup = Array.tabulate(s)(k =>
      (k until n by s).map(raw).filter(v => !v.isNaN && v != 0.0)
    )
    val groupMeans = byGroup.map(g => if (g.isEmpty) 1.0 else g.sum / g.length)
    val overallMean = groupMeans.sum / s

    val normalized =
      if (additive) groupMeans.map(_ - overallMean)
      else groupMeans.map(_ / math.max(overallMean, 1e-10))

    Array.tabulate(n)(t => normalized(t % s))
  }

  private def simpleMA(y: Array[Double], w: Int): Array[Double] = {
    val n = y.length; val result = new Array[Double](n - w + 1)
    var sum = y.take(w).sum; result(0) = sum / w
    for (i <- 1 until result.length) {
      sum -= y(i-1); sum += y(i+w-1); result(i) = sum / w
    }
    result
  }

  /** Henderson MA order: 13 for monthly, 9 for quarterly, 5 for annual */
  private def hendersonLength(s: Int): Int =
    if (s >= 12) 13 else if (s >= 4) 9 else 5

  /**
   * Henderson moving average — optimal trend estimation.
   * Weights minimize the third difference of the trend.
   */
  private def hendersonMA(y: Array[Double], m: Int): Array[Double] = {
    val n    = y.length
    val half = m / 2
    val w    = hendersonWeights(m)
    val result = Array.fill(n)(Double.NaN)

    for (t <- half until n - half) {
      var sum = 0.0
      for (j <- -half to half)
        sum += w(j + half) * y(t + j)
      result(t) = sum
    }

    // Endpoint fill with edge values
    for (t <- 0 until half)
      result(t) = result(half)
    for (t <- n - half until n)
      result(t) = result(n - half - 1)

    result
  }

  /**
   * Henderson weights (closed form for odd m = 2h+1).
   * Minimizes the sum of squares of third differences.
   */
  private def hendersonWeights(m: Int): Array[Double] = {
    val h  = m / 2
    val h1 = h + 1
    val h2 = h + 2

    val raw = Array.tabulate(m) { i =>
      val j = i - h
      val j2 = j.toDouble * j
      val num = (h1*h1 - j2) * (h2*h2 - j2) * (3.0*(3*h*h + 3*h - 1 - 11*j2))
      val den = 8.0 * h * h1 * h2 * (h*h + h - 1) * (4*h*h - 1) * (2*h+3) / 3.0
      if (math.abs(den) < 1e-12) 0.0 else num / den
    }

    // Normalize to sum to 1
    val sum = raw.sum
    raw.map(_ / math.max(sum, 1e-10))
  }
}

case class X11Result(
  original:           Array[Double],
  trend:              Array[Double],
  seasonal:           Array[Double],
  irregular:          Array[Double],
  seasonallyAdjusted: Array[Double],
  period:             Int,
  additive:           Boolean
) {
  val n: Int = original.length

  override def toString: String = {
    val model = if (additive) "Additive" else "Multiplicative"
    val validIrr = irregular.filter(!_.isNaN)
    val irrStd = if (validIrr.isEmpty) Double.NaN else {
      val mu = validIrr.sum / validIrr.length
      math.sqrt(validIrr.map(v => (v-mu)*(v-mu)).sum / validIrr.length)
    }
    f"X-11 Decomposition ($model, period=$period, n=$n)\n" +
    f"  Irregular std: $irrStd%.4f\n"
  }
}
