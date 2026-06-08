package series.filters

/**
 * Christiano-Fitzgerald band-pass filter.
 *
 * Unlike Baxter-King which uses fixed symmetric weights,
 * the CF filter uses time-varying (observation-dependent) weights
 * that are optimal in a mean-square sense for each observation.
 *
 * Two variants:
 *   Symmetric  (default) — uses full sample, optimal for offline analysis
 *   Asymmetric           — real-time one-sided approximation
 *
 * The CF filter does NOT drop endpoint observations (unlike BK),
 * but the symmetric version requires the full series length
 * in the weight computation.
 *
 * Standard settings (quarterly):
 *   pl = 6, ph = 32
 *
 * @param pl   lower period bound
 * @param ph   upper period bound
 */
class ChristianoFitzgeraldFilter(
  pl: Int = 6,
  ph: Int = 32
) {
  require(pl >= 2, "lower period bound must be at least 2")
  require(ph > pl, "upper period must exceed lower period")

  private val omegaL = 2.0 * math.Pi / ph
  private val omegaH = 2.0 * math.Pi / pl

  /**
   * Ideal high-pass weight at lag k.
   * B(k) = sin(omega*k)/(pi*k) for k!=0, omega/pi for k=0
   */
  private def b(omega: Double, k: Int): Double =
    if (k == 0) omega / math.Pi
    else math.sin(omega * k) / (math.Pi * k)

  /**
   * Applies the symmetric CF filter using time-varying optimal weights.
   *
   * For each time t, the weight vector is computed using all available
   * leads and lags up to the series boundaries, with endpoint
   * corrections to ensure the weights sum to zero.
   *
   * @param y input series (length n >= 4)
   * @return CFFilterResult
   */
  def filterSymmetric(y: Array[Double]): CFFilterResult = {
    val n = y.length
    require(n >= 4, "series must have at least 4 observations")

    val filtered = Array.tabulate(n) { t =>
      // Optimal weights for observation t
      // B_t(k) = b(omegaH, k) - b(omegaL, k) for k in {-(t-1)..n-t-1}
      // with endpoint correction
      val maxBack = t
      val maxFwd  = n - 1 - t

      // Raw band-pass weights
      val rawBP = ((-maxBack) to maxFwd).map { k =>
        b(omegaH, k) - b(omegaL, k)
      }.toArray

      // Endpoint correction: adjust first and last weight so sum = 0
      // Following CF (1999): distribute residual to endpoints
      val sumRaw = rawBP.sum
      if (rawBP.length >= 2) {
        rawBP(0)              -= sumRaw / 2.0
        rawBP(rawBP.length-1) -= sumRaw / 2.0
      }

      // Apply to data
      rawBP.zipWithIndex.map { case (w, j) =>
        w * y(t - maxBack + j)
      }.sum
    }

    CFFilterResult(
      filtered  = filtered,
      symmetric = true,
      pl        = pl,
      ph        = ph
    )
  }

  /**
   * Applies the asymmetric CF filter (one-sided, real-time approximation).
   *
   * Uses only past observations at each point (no future values).
   * Weights are the one-sided truncation of the ideal filter,
   * adjusted so they sum to zero.
   *
   * @param y     input series
   * @param maxLag maximum lags to use (defaults to series length - 1)
   * @return CFFilterResult
   */
  def filterAsymmetric(
    y:      Array[Double],
    maxLag: Int = -1
  ): CFFilterResult = {
    val n   = y.length
    val lag = if (maxLag < 0) n - 1 else math.min(maxLag, n - 1)
    require(n >= 2, "series must have at least 2 observations")

    val filtered = Array.tabulate(n) { t =>
      val kMax = math.min(t, lag)
      val raw  = (0 to kMax).map(k => b(omegaH, k) - b(omegaL, k)).toArray
      // Adjust so weights sum to zero
      val sumRaw = raw.sum
      if (raw.length >= 1) raw(0) -= sumRaw
      raw.zipWithIndex.map { case (w, k) => w * y(t - k) }.sum
    }

    CFFilterResult(
      filtered  = filtered,
      symmetric = false,
      pl        = pl,
      ph        = ph
    )
  }
}

case class CFFilterResult(
  filtered:  Array[Double],
  symmetric: Boolean,
  pl:        Int,
  ph:        Int
) {
  def cycleStd: Double = {
    val mu = filtered.sum / filtered.length
    math.sqrt(filtered.map(f => math.pow(f - mu, 2)).sum / filtered.length)
  }

  override def toString: String =
    f"Christiano-Fitzgerald Filter (pl=$pl, ph=$ph, " +
    f"${if (symmetric) "symmetric" else "asymmetric"})\n" +
    f"  Output length: ${filtered.length}\n" +
    f"  Cycle std:     $cycleStd%.4f\n"
}
