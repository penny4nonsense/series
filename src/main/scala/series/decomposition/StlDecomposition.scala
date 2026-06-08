package series.decomposition

/**
 * STL (Seasonal-Trend decomposition using LOESS).
 * Cleveland, Cleveland, McRae, Terpenning (1990).
 *
 * Additive decomposition: Y_t = T_t + S_t + R_t
 *
 * @param period  seasonal period s
 * @param ns      seasonal LOESS window (odd, >= 7; -1 = auto)
 * @param nt      trend LOESS window (odd; -1 = auto)
 * @param nl      low-pass LOESS window (odd >= s; -1 = s)
 * @param nInner  inner loop iterations (default 2)
 * @param robust  use bisquare robustness weights (default true)
 */
class StlDecomposition(
  period: Int,
  ns:     Int     = -1,
  nt:     Int     = -1,
  nl:     Int     = -1,
  nInner: Int     = 2,
  robust: Boolean = true
) {
  require(period >= 2, "period must be at least 2")

  private val nOuter   = if (robust) 2 else 0
  private val nsWindow = if (ns < 0) nextOdd(math.max(7, period)) else ns
  private val ntWindow = if (nt < 0) nextOdd(math.max(3, (1.5 * period / (1.0 - 1.5 / nsWindow) + 0.5).toInt)) else nt
  private val nlWindow = if (nl < 0) nextOdd(period) else nl

  private def nextOdd(x: Int): Int = if (x % 2 == 1) x else x + 1

  def decompose(y: Array[Double]): StlResult = {
    val n = y.length
    require(n >= 2 * period, s"series too short: $n < ${2*period}")

    var trend    = Array.fill(n)(0.0)
    var seasonal = Array.fill(n)(0.0)
    var weights  = Array.fill(n)(1.0)

    for (outer <- 0 to nOuter) {
      for (_ <- 0 until nInner) {
        // Step 1: Detrend
        val detrended = Array.tabulate(n)(t => y(t) - trend(t))

        // Step 2: Cycle-subseries smoothing — returns length-n array
        val csSmoothed = smoothCycleSubseries(detrended, n, weights)

        // Step 3: Low-pass filter of the cycle-subseries
        val lp = lowPassSmooth(csSmoothed, n)

        // Step 4: Seasonal = cycle-smoothed - low-pass
        seasonal = Array.tabulate(n)(t => csSmoothed(t) - lp(t))

        // Step 5: Deseasonalize and smooth for trend
        val deseas = Array.tabulate(n)(t => y(t) - seasonal(t))
        trend = loess(deseas, Array.fill(n)(1.0), ntWindow)
      }

      if (outer < nOuter) {
        val rem = Array.tabulate(n)(t => y(t) - trend(t) - seasonal(t))
        weights = bisquareWeights(rem)
      }
    }

    val remainder = Array.tabulate(n)(t => y(t) - trend(t) - seasonal(t))

    StlResult(
      original  = y,
      trend     = trend,
      seasonal  = seasonal,
      remainder = remainder,
      period    = period,
      robust    = robust,
      weights   = if (robust) Some(bisquareWeights(remainder)) else None
    )
  }

  /**
   * For each season k=0..period-1, LOESS-smooth the subseries
   * y(k), y(k+period), y(k+2*period), ...
   * Returns a full-length array with smoothed values placed back.
   * Uses linear extrapolation to fill one extra point at each end
   * so the low-pass filter can operate on the full range.
   */
  private def smoothCycleSubseries(
    y:       Array[Double],
    n:       Int,
    weights: Array[Double]
  ): Array[Double] = {
    // Extended: one extra season on each end
    val extLen = n + 2 * period
    val ext    = new Array[Double](extLen)

    for (k <- 0 until period) {
      val subIdx  = (k until n by period).toArray
      val m       = subIdx.length
      val subY    = subIdx.map(y)
      val subW    = subIdx.map(weights)
      val subX    = Array.tabulate(m)(i => i.toDouble)
      val sm      = loess1d(subX, subY, subW, nsWindow)

      // Place smoothed values at positions k + (i+1)*period in extended array
      for (i <- 0 until m)
        ext(k + (i + 1) * period) = sm(i)

      // Extrapolate one step before and after
      ext(k) = if (m >= 2) 2*sm(0) - sm(1) else sm(0)
      ext(k + (m + 1) * period) = if (m >= 2) 2*sm(m-1) - sm(m-2) else sm(m-1)
    }

    // Extract center n values (offset by period)
    Array.tabulate(n)(t => ext(t + period))
  }

  /**
   * Low-pass filter: MA(s) → MA(s) → MA(3) → LOESS(nlWindow)
   * All applied to the length-n cycle-subseries output.
   */
  private def lowPassSmooth(cs: Array[Double], n: Int): Array[Double] = {
    val ma1 = centeredMA(cs, period)
    val ma2 = centeredMA(ma1, period)
    val ma3 = centeredMA(ma2, 3)
    // ma3 may be shorter — pad with edge values to length n
    val padded = padToLength(ma3, n)
    loess(padded, Array.fill(n)(1.0), nlWindow)
  }

  /** Simple moving average (output shorter than input) */
  private def centeredMA(y: Array[Double], w: Int): Array[Double] = {
    val n      = y.length
    val outLen = n - w + 1
    if (outLen <= 0) return y
    val result = new Array[Double](outLen)
    var sum    = y.take(w).sum
    result(0)  = sum / w
    for (i <- 1 until outLen) {
      sum -= y(i - 1); sum += y(i + w - 1)
      result(i) = sum / w
    }
    result
  }

  /** Pad array to target length by repeating edge values */
  private def padToLength(y: Array[Double], target: Int): Array[Double] = {
    if (y.length == target) return y
    val result = new Array[Double](target)
    val diff   = target - y.length
    val left   = diff / 2
    for (i <- 0 until target) {
      val j = math.max(0, math.min(y.length - 1, i - left))
      result(i) = y(j)
    }
    result
  }

  /** LOESS on regularly-spaced data (x = 0, 1, ..., n-1) */
  private def loess(y: Array[Double], w: Array[Double], q: Int): Array[Double] = {
    val x = Array.tabulate(y.length)(_.toDouble)
    loess1d(x, y, w, q)
  }

  /**
   * LOESS with arbitrary x-coordinates.
   * Tricube kernel, degree-1 polynomial, q nearest neighbors.
   */
  private def loess1d(
    x: Array[Double],
    y: Array[Double],
    w: Array[Double],
    q: Int
  ): Array[Double] = {
    val n      = x.length
    val result = new Array[Double](n)
    val qEff   = math.min(math.max(q, 2), n)

    for (i <- 0 until n) {
      val dists   = x.map(xi => math.abs(xi - x(i)))
      val sorted  = dists.sorted
      val maxD    = math.max(sorted(qEff - 1), 1e-10)

      val wi = Array.tabulate(n) { j =>
        val u = dists(j) / maxD
        if (u >= 1.0) 0.0 else math.pow(1.0 - u*u*u, 3) * w(j)
      }

      val sw   = wi.sum
      val swx  = (0 until n).map(j => wi(j) * x(j)).sum
      val swx2 = (0 until n).map(j => wi(j) * x(j) * x(j)).sum
      val swy  = (0 until n).map(j => wi(j) * y(j)).sum
      val swxy = (0 until n).map(j => wi(j) * x(j) * y(j)).sum

      val det = sw * swx2 - swx * swx
      result(i) = if (math.abs(det) < 1e-12) {
        if (sw > 1e-10) swy / sw else y(i)
      } else {
        val b = (sw * swxy - swx * swy) / det
        val a = (swy - b * swx) / sw
        a + b * x(i)
      }
    }
    result
  }

  private def bisquareWeights(r: Array[Double]): Array[Double] = {
    val absR = r.map(math.abs)
    val med  = median(absR)
    val h    = 6.0 * math.max(med, 1e-10)
    r.map { ri =>
      val u = math.abs(ri) / h
      if (u >= 1.0) 0.0 else math.pow(1.0 - u*u, 2)
    }
  }

  private def median(x: Array[Double]): Double = {
    val s = x.sorted; val n = s.length
    if (n % 2 == 1) s(n/2) else (s(n/2-1) + s(n/2)) / 2.0
  }
}

case class StlResult(
  original:  Array[Double],
  trend:     Array[Double],
  seasonal:  Array[Double],
  remainder: Array[Double],
  period:    Int,
  robust:    Boolean,
  weights:   Option[Array[Double]]
) {
  val n: Int = original.length

  private def variance(x: Array[Double]): Double = {
    val mu = x.sum / x.length
    x.map(v => (v-mu)*(v-mu)).sum / x.length
  }

  def seasonalStrength: Double = {
    val varR  = variance(remainder)
    val varSR = variance(seasonal.zip(remainder).map { case (a,b) => a+b })
    math.max(0.0, 1.0 - varR / math.max(varSR, 1e-10))
  }

  def trendStrength: Double = {
    val varR  = variance(remainder)
    val varTR = variance(trend.zip(remainder).map { case (a,b) => a+b })
    math.max(0.0, 1.0 - varR / math.max(varTR, 1e-10))
  }

  def seasonalSumCheck: Double = {
    val nFull = (n / period) * period
    seasonal.take(nFull).sum / (n / period)
  }

  override def toString: String = {
    val robustStr = if (robust) "robust" else "non-robust"
    f"STL Decomposition ($robustStr, period=$period, n=$n)\n" +
    f"  Seasonal strength: $seasonalStrength%.4f\n" +
    f"  Trend strength:    $trendStrength%.4f\n" +
    f"  Seasonal sum/period: $seasonalSumCheck%.6f\n"
  }
}
