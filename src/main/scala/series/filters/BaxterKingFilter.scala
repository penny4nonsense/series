package series.filters

/**
 * Baxter-King band-pass filter.
 *
 * Extracts business cycle components by applying a symmetric moving average
 * with weights designed to pass frequencies in the band [2*pi/pl, 2*pi/ph]
 * (periods between pl and ph quarters/months).
 *
 * Standard settings (quarterly data):
 *   pl = 6, ph = 32 (business cycles: 1.5 to 8 years)
 *   K  = 12 (number of leads/lags, so filter length = 2K+1)
 *
 * The filter loses K observations at each end.
 *
 * Two variants:
 *   Symmetric  — uses full sample, optimal for analysis (default)
 *   Asymmetric — one-sided approximation for real-time use
 *
 * @param pl   lower period bound (passes cycles longer than pl)
 * @param ph   upper period bound (passes cycles shorter than ph)
 * @param K    number of leads/lags (filter half-length)
 */
class BaxterKingFilter(
  pl: Int    = 6,
  ph: Int    = 32,
  K:  Int    = 12
) {
  require(pl >= 2,  "lower period bound must be at least 2")
  require(ph > pl,  "upper period must exceed lower period")
  require(K  >= 1,  "K must be at least 1")

  private val omegaL = 2.0 * math.Pi / ph   // lower cutoff frequency
  private val omegaH = 2.0 * math.Pi / pl   // upper cutoff frequency

  /**
   * Ideal band-pass filter weights (before endpoint adjustment).
   * b_k = (sin(omegaH*k) - sin(omegaL*k)) / (pi*k)  for k != 0
   * b_0 = (omegaH - omegaL) / pi
   */
  private val rawWeights: Array[Double] = {
    val w = new Array[Double](2 * K + 1)
    w(K) = (omegaH - omegaL) / math.Pi  // b_0
    for (k <- 1 to K) {
      val bk = (math.sin(omegaH * k) - math.sin(omegaL * k)) / (math.Pi * k)
      w(K + k) = bk
      w(K - k) = bk
    }
    w
  }

  /**
   * Adjusted weights: subtract a constant from each weight so that
   * the weights sum to zero (exact removal of unit roots).
   * theta = mean of raw weights
   */
  val weights: Array[Double] = {
    val theta = rawWeights.sum / rawWeights.length
    rawWeights.map(_ - theta)
  }

  /**
   * Applies the symmetric BK filter.
   * Loses K observations at each end.
   *
   * @param y input series (length n, requires n > 2K)
   * @return BKFilterResult with filtered series and endpoints
   */
  def filterSymmetric(y: Array[Double]): BKFilterResult = {
    val n = y.length
    require(n > 2 * K, s"series length $n must exceed 2*K = ${2*K}")

    val filtered = Array.tabulate(n - 2 * K) { t =>
      val tFull = t + K
      (0 until weights.length).map(j => weights(j) * y(tFull - K + j)).sum
    }

    BKFilterResult(
      filtered   = filtered,
      nDropped   = K,
      symmetric  = true,
      pl         = pl,
      ph         = ph,
      K          = K,
      weights    = weights
    )
  }

  /**
   * Applies an asymmetric one-sided approximation.
   * Uses available leads/lags at endpoints — fewer lags used
   * at the start, fewer leads at the end.
   * No observations are dropped but end quality is lower.
   *
   * @param y input series
   * @return BKFilterResult with full-length filtered series
   */
  def filterAsymmetric(y: Array[Double]): BKFilterResult = {
    val n = y.length
    require(n >= K + 1, s"series length $n must exceed K = $K")

    val filtered = Array.tabulate(n) { t =>
      val kBack  = math.min(t,     K)
      val kFwd   = math.min(n-1-t, K)
      // Renormalize available weights
      val avail  = weights.slice(K - kBack, K + kFwd + 1)
      val sum    = avail.sum
      val norm   = if (math.abs(sum) > 1e-10) avail.map(_ / sum) else avail
      norm.zipWithIndex.map { case (w, j) => w * y(t - kBack + j) }.sum
    }

    BKFilterResult(
      filtered   = filtered,
      nDropped   = 0,
      symmetric  = false,
      pl         = pl,
      ph         = ph,
      K          = K,
      weights    = weights
    )
  }
}

case class BKFilterResult(
  filtered:  Array[Double],
  nDropped:  Int,
  symmetric: Boolean,
  pl:        Int,
  ph:        Int,
  K:         Int,
  weights:   Array[Double]
) {
  def cycleStd: Double = {
    val mu = filtered.sum / filtered.length
    math.sqrt(filtered.map(f => math.pow(f - mu, 2)).sum / filtered.length)
  }

  override def toString: String =
    f"Baxter-King Filter (pl=$pl, ph=$ph, K=$K, " +
    f"${if (symmetric) "symmetric" else "asymmetric"})\n" +
    f"  Output length: ${filtered.length} (dropped $nDropped at each end)\n" +
    f"  Cycle std:     $cycleStd%.4f\n" +
    f"  Weights sum:   ${weights.sum%.6f} (should be ~0)\n"
}
