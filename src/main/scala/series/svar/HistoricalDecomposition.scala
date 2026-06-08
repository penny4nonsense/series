package series.svar

import breeze.linalg._

/**
 * Historical decomposition of observed time series into
 * contributions from each structural shock.
 *
 * Decomposes the observed deviation from the unconditional mean
 * into the cumulative contribution of each structural shock:
 *
 *   y_t - mu = sum_j sum_s Theta_s(j) * epsilon_{t-s,j}
 *
 * where Theta_s(j) is the IRF of y to shock j at horizon s
 * and epsilon_{t,j} are the structural shocks.
 *
 * For sign-restricted models uses the median impact matrix.
 */
object HistoricalDecomposition {

  /**
   * Computes historical decomposition for a point-identified SVAR.
   *
   * @param result SvarResult from point identification
   * @return HistoricalDecompositionResult
   */
  def decompose(result: SvarResult): HistoricalDecompositionResult = {
    val k      = result.k
    val shocks = result.shocks  // (n-p x k)
    val n      = shocks.rows
    val sirfs  = result.sirfs

    // Contribution of shock j to series i at time t
    // = sum_{s=0}^{t} Theta_s(i,j) * epsilon_{t-s, j}
    val contributions = Array.tabulate(n, k, k) { (t, i, j) =>
      (0 to math.min(t, result.h)).map { s =>
        if (s < sirfs.length)
          sirfs(s)(i, j) * shocks(t - s, j)
        else 0.0
      }.sum
    }

    HistoricalDecompositionResult(
      contributions = contributions,
      shocks        = shocks,
      n             = n,
      k             = k,
      shockNames    = result.shockNames
    )
  }

  /**
   * Computes historical decomposition for sign-restricted SVAR
   * using the median impact matrix.
   *
   * @param result SignRestrictionResult
   * @param varFit underlying VarFit for reduced form residuals
   * @return HistoricalDecompositionResult using median model
   */
  def decompose(
                 result: SignRestrictionResult,
                 model:  SvarModel
               ): HistoricalDecompositionResult = {
    // Use median B0inv across admissible draws
    val k = result.k

    val medianB0inv = DenseMatrix.tabulate(k, k) { (i, j) =>
      val vals = result.admissibleB0invs.map(_(i, j)).sorted
      vals(vals.length / 2)
    }

    val shocks = model.structuralShocks(medianB0inv)
    val n      = shocks.rows

    // Recompute IRFs with median B0inv
    val sirfs = model.structuralIRF(medianB0inv, result.h)

    val contributions = Array.tabulate(n, k, k) { (t, i, j) =>
      (0 to math.min(t, result.h)).map { s =>
        if (s < sirfs.length)
          sirfs(s)(i, j) * shocks(t - s, j)
        else 0.0
      }.sum
    }

    HistoricalDecompositionResult(
      contributions = contributions,
      shocks        = shocks,
      n             = n,
      k             = k,
      shockNames    = result.shockNames
    )
  }
}

/**
 * Historical decomposition results.
 *
 * @param contributions array (n x k x k)
 *                      contributions(t)(i)(j) = contribution of shock j
 *                      to series i at time t
 * @param shocks        structural shock series (n x k)
 * @param n             number of time periods
 * @param k             number of series
 * @param shockNames    names of structural shocks
 */
case class HistoricalDecompositionResult(
                                          contributions: Array[Array[Array[Double]]],
                                          shocks:        DenseMatrix[Double],
                                          n:             Int,
                                          k:             Int,
                                          shockNames:    Array[String]
                                        ) {
  /**
   * Total contribution of shock j to series i over all time.
   */
  def totalContribution(series: Int, shock: Int): Array[Double] =
    (0 until n).map(t => contributions(t)(series)(shock)).toArray

  /**
   * Fraction of variance of series i explained by shock j.
   */
  def varianceShare(series: Int, shock: Int): Double = {
    val total = (0 until n).map(t =>
      (0 until k).map(j => contributions(t)(series)(j) *
        contributions(t)(series)(j)).sum
    ).sum
    val byShock = (0 until n).map(t =>
      contributions(t)(series)(shock) * contributions(t)(series)(shock)
    ).sum
    if (total > 0) byShock / total else 0.0
  }

  override def toString: String = {
    val sb = new StringBuilder
    sb.append("Historical Decomposition:\n\n")
    for (i <- 0 until k) {
      sb.append(s"  Series y${i+1}:\n")
      sb.append(f"  ${"t"}%5s" +
        shockNames.map(n => f"  $n%10s").mkString + "\n")
      for (t <- 0 until math.min(n, 20))
        sb.append(f"  $t%5d" +
          (0 until k).map(j =>
            f"  ${contributions(t)(i)(j)}%10.4f").mkString + "\n")
      sb.append("\n")

      sb.append("  Variance shares:\n")
      for (j <- 0 until k)
        sb.append(f"  ${shockNames(j)}%-15s  " +
          f"${varianceShare(i, j) * 100}%.1f%%\n")
      sb.append("\n")
    }
    sb.toString
  }
}