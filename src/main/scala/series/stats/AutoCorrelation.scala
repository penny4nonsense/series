package series.stats

import breeze.linalg._

/**
 * Autocorrelation and partial autocorrelation functions.
 *
 * ACF measures the correlation between a series and its own lags.
 * PACF measures the correlation at lag k after removing the effect
 * of all intermediate lags — computed via the Durbin-Levinson
 * algorithm which is exact and numerically stable.
 *
 * Both are fundamental tools for ARIMA order identification:
 *   - AR(p): ACF tails off, PACF cuts off at lag p
 *   - MA(q): ACF cuts off at lag q, PACF tails off
 *   - ARMA:  both tail off
 */
object AutoCorrelation {

  /**
   * Computes the sample autocorrelation function up to maxLag.
   *
   * @param y      time series
   * @param maxLag maximum lag
   * @return array of autocorrelations at lags 1..maxLag
   */
  def acf(y: Array[Double], maxLag: Int): Array[Double] = {
    require(maxLag > 0, "maxLag must be positive")
    require(maxLag < y.length, "maxLag must be less than series length")

    val n  = y.length
    val mu = y.sum / n
    val v  = y.map(_ - mu)
    val c0 = v.map(x => x * x).sum / n

    (1 to maxLag).map { lag =>
      val c = (0 until n - lag).map(i => v(i) * v(i + lag)).sum / n
      c / c0
    }.toArray
  }

  /**
   * Computes the sample partial autocorrelation function up to maxLag
   * using the Durbin-Levinson algorithm.
   *
   * The Durbin-Levinson recursion computes PACF exactly from the ACF
   * without fitting separate AR models at each lag — O(maxLag^2) and
   * numerically stable.
   *
   * @param y      time series
   * @param maxLag maximum lag
   * @return array of partial autocorrelations at lags 1..maxLag
   */
  def pacf(y: Array[Double], maxLag: Int): Array[Double] = {
    require(maxLag > 0, "maxLag must be positive")
    require(maxLag < y.length, "maxLag must be less than series length")

    val r    = Array(1.0) ++ acf(y, maxLag)  // r(0)=1, r(1..maxLag)=ACF
    val pacf = Array.fill(maxLag)(0.0)

    // Durbin-Levinson recursion
    // phi(k,k) is the PACF at lag k
    // phi(k,j) are the intermediate AR coefficients
    var phi = Array(r(1))  // phi(1,1) = r(1)
    pacf(0) = r(1)

    for (k <- 2 to maxLag) {
      // Compute phi(k,k)
      val num   = r(k) - (1 until k).map(j => phi(j-1) * r(k-j)).sum
      val denom = 1.0  - (1 until k).map(j => phi(j-1) * r(j)).sum
      val phikk = if (math.abs(denom) < 1e-10) 0.0 else num / denom
      pacf(k-1) = phikk

      // Update phi coefficients
      val phiNew = Array.fill(k)(0.0)
      for (j <- 1 until k)
        phiNew(j-1) = phi(j-1) - phikk * phi(k-j-1)
      phiNew(k-1) = phikk
      phi = phiNew
    }

    pacf
  }

  /**
   * Bartlett's formula for 95% confidence bands on ACF.
   * Under H0 of white noise, ACF(k) ~ N(0, 1/n) approximately.
   *
   * @param n series length
   * @return 95% confidence bound (symmetric around zero)
   */
  def confidenceBound(n: Int): Double = 1.96 / math.sqrt(n)

  /**
   * Identifies lags where ACF exceeds the confidence bound.
   * Useful for MA order identification.
   *
   * @param y      time series
   * @param maxLag maximum lag to check
   * @return indices of significant lags (1-based)
   */
  def significantACFLags(y: Array[Double], maxLag: Int): Array[Int] = {
    val bound = confidenceBound(y.length)
    val a     = acf(y, maxLag)
    a.zipWithIndex
      .filter { case (v, _) => math.abs(v) > bound }
      .map { case (_, i) => i + 1 }
  }

  /**
   * Identifies lags where PACF exceeds the confidence bound.
   * Useful for AR order identification.
   *
   * @param y      time series
   * @param maxLag maximum lag to check
   * @return indices of significant lags (1-based)
   */
  def significantPACFLags(y: Array[Double], maxLag: Int): Array[Int] = {
    val bound = confidenceBound(y.length)
    val p     = pacf(y, maxLag)
    p.zipWithIndex
      .filter { case (v, _) => math.abs(v) > bound }
      .map { case (_, i) => i + 1 }
  }

  /**
   * Suggests tentative ARIMA orders based on ACF/PACF cutoffs.
   * This is a heuristic — always verify with AIC/BIC and diagnostics.
   *
   * @param y      time series
   * @param maxLag maximum lag to consider
   * @return OrderSuggestion with tentative p and q
   */
  def suggestOrders(y: Array[Double], maxLag: Int): OrderSuggestion = {
    val bound     = confidenceBound(y.length)
    val acfVals   = acf(y, maxLag)
    val pacfVals  = pacf(y, maxLag)

    // Find cutoff lags — last significant lag after which all are small
    def cutoffLag(vals: Array[Double]): Option[Int] = {
      val sig = vals.zipWithIndex.filter { case (v, _) => math.abs(v) > bound }
      if (sig.isEmpty) None
      else Some(sig.last._2 + 1)
    }

    val acfCutoff  = cutoffLag(acfVals)
    val pacfCutoff = cutoffLag(pacfVals)

    (acfCutoff, pacfCutoff) match {
      case (None, None)       => OrderSuggestion(p=0, q=0, note="White noise")
      case (Some(q), None)    => OrderSuggestion(p=0, q=q, note="MA process suggested")
      case (None, Some(p))    => OrderSuggestion(p=p, q=0, note="AR process suggested")
      case (Some(_), Some(_)) => OrderSuggestion(p=1, q=1, note="ARMA process suggested — use AIC/BIC for order selection")
    }
  }
}

/**
 * Tentative order suggestion from ACF/PACF analysis.
 *
 * @param p  suggested AR order
 * @param q  suggested MA order
 * @param note interpretation note
 */
case class OrderSuggestion(p: Int, q: Int, note: String) {
  override def toString: String =
    s"Suggested ARIMA($p, ?, $q) — $note"
}