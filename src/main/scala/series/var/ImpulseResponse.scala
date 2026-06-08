package series.`var`

import breeze.linalg._

/**
 * Impulse response functions and forecast error variance decomposition
 * for fitted VAR models.
 *
 * IRFs trace the effect of a one-unit shock to series j on series i
 * over h periods. Two identification schemes are supported:
 *
 *   Orthogonal (Cholesky) — shocks are orthogonalized via Cholesky
 *   decomposition of Sigma. Ordering matters.
 *
 *   Generalized (Pesaran-Shin) — ordering-invariant but shocks
 *   are not mutually orthogonal.
 *
 * FEVD decomposes the h-step forecast error variance of each series
 * into contributions from each structural shock.
 *
 * @param fit fitted VarFit
 */
class ImpulseResponse(fit: VarFit) {

  private val k = fit.k
  private val C = fit.companionMatrix

  /**
   * Cholesky factor of Sigma for orthogonal identification.
   * Lower triangular P such that P * P' = Sigma.
   */
  private val P: DenseMatrix[Double] = cholesky(fit.Sigma)

  /**
   * Computes moving average coefficient matrices Phi_0, Phi_1, ..., Phi_h.
   * Phi_s gives the response at horizon s to a shock at time 0.
   *
   * Uses the recursion:
   *   Phi_s = A_1 * Phi_{s-1} + ... + A_p * Phi_{s-p}
   * with Phi_0 = I_k, Phi_s = 0 for s < 0.
   */
  def maCoefficients(h: Int): Array[DenseMatrix[Double]] = {
    val phis = Array.fill(h + 1)(DenseMatrix.zeros[Double](k, k))
    phis(0) = DenseMatrix.eye[Double](k)

    for (s <- 1 to h) {
      for (lag <- 1 to math.min(s, fit.p)) {
        val A    = fit.lagMatrix(lag).copy
        val prod = A * phis(s - lag).copy
        phis(s) += prod
      }
    }
    phis
  }

  /**
   * Orthogonalized impulse response functions via Cholesky decomposition.
   *
   * @param h      forecast horizon
   * @return IRFResult with responses (h+1 x k x k)
   *         result(s)(i)(j) = response of series i at horizon s
   *         to a one-std shock in series j
   */
  def orthogonalIRF(h: Int): IRFResult = {
    require(h > 0, "horizon must be positive")
    val phis = maCoefficients(h)

    // Orthogonalized responses: Theta_s = Phi_s * P
    val responses = phis.map(_ * P)

    IRFResult(responses, k, h, orthogonal = true)
  }

  /**
   * Generalized impulse response functions (Pesaran & Shin 1998).
   * Ordering-invariant alternative to Cholesky.
   *
   * @param h forecast horizon
   * @return IRFResult with generalized responses
   */
  def generalizedIRF(h: Int): IRFResult = {
    require(h > 0, "horizon must be positive")
    val phis   = maCoefficients(h)
    val SigInv = inv(fit.Sigma)

    // Generalized response of series i to shock in series j:
    // GIRF(i,j,s) = (sigma_jj)^{-0.5} * e_i' * Phi_s * Sigma * e_j
    val responses = phis.map { phi =>
      val resp = DenseMatrix.zeros[Double](k, k)
      for (i <- 0 until k; j <- 0 until k) {
        val ei    = DenseVector.zeros[Double](k); ei(i) = 1.0
        val ej    = DenseVector.zeros[Double](k); ej(j) = 1.0
        val sigJJ = fit.Sigma(j, j)
        resp(i, j) = (ei.t * phi * fit.Sigma * ej) / math.sqrt(sigJJ)
      }
      resp
    }

    IRFResult(responses, k, h, orthogonal = false)
  }

  /**
   * Forecast error variance decomposition.
   * Decomposes variance of h-step forecast error of series i
   * into contributions from each shock j.
   *
   * Uses orthogonalized IRFs.
   *
   * @param h forecast horizon
   * @return FEVDResult (h+1 x k x k) proportions summing to 1
   */
  def fevd(h: Int): FEVDResult = {
    require(h > 0, "horizon must be positive")
    val irf   = orthogonalIRF(h)

    // MSE matrix for each horizon
    // MSE_i(h) = sum_{s=0}^{h} sum_j theta_{ij,s}^2
    val contributions = Array.tabulate(h + 1, k, k) { (s, i, j) =>
      irf.responses(s)(i, j) * irf.responses(s)(i, j)
    }

    // Cumulative sum over horizons
    val cumulative = Array.tabulate(h + 1, k, k) { (s, i, j) =>
      (0 to s).map(t => contributions(t)(i)(j)).sum
    }

    // Normalize to proportions
    val proportions = Array.tabulate(h + 1, k, k) { (s, i, j) =>
      val total = (0 until k).map(jj => cumulative(s)(i)(jj)).sum
      if (total > 0) cumulative(s)(i)(j) / total else 0.0
    }

    FEVDResult(proportions, k, h)
  }
}

/**
 * Impulse response function results.
 *
 * @param responses  array of (h+1) response matrices (k x k)
 *                   responses(s)(i,j) = response of i at horizon s to shock in j
 * @param k          number of series
 * @param h          forecast horizon
 * @param orthogonal whether shocks are orthogonalized
 */
case class IRFResult(
                      responses:   Array[DenseMatrix[Double]],
                      k:           Int,
                      h:           Int,
                      orthogonal:  Boolean
                    ) {
  /**
   * Response of series `toSeries` to shock in `fromSeries`
   * over all horizons.
   */
  def response(fromSeries: Int, toSeries: Int): Array[Double] =
    responses.map(_(toSeries, fromSeries))

  override def toString: String = {
    val sb   = new StringBuilder
    val kind = if (orthogonal) "Orthogonalized" else "Generalized"
    sb.append(s"$kind Impulse Response Functions:\n\n")
    for (j <- 0 until k) {
      sb.append(s"  Shock to series ${j+1}:\n")
      sb.append(f"  ${"h"}%5s" + (0 until k).map(i => f"  y${i+1}%8s").mkString + "\n")
      for (s <- 0 to h)
        sb.append(f"  $s%5d" + (0 until k).map(i =>
          f"  ${responses(s)(i,j)}%9.4f").mkString + "\n")
      sb.append("\n")
    }
    sb.toString
  }
}

/**
 * Forecast error variance decomposition results.
 *
 * @param proportions array of (h+1 x k x k) proportions
 *                    proportions(s)(i)(j) = fraction of variance of series i
 *                    at horizon s due to shock in series j
 * @param k           number of series
 * @param h           forecast horizon
 */
case class FEVDResult(
                       proportions: Array[Array[Array[Double]]],
                       k:           Int,
                       h:           Int
                     ) {
  override def toString: String = {
    val sb = new StringBuilder
    sb.append("Forecast Error Variance Decomposition:\n\n")
    for (i <- 0 until k) {
      sb.append(s"  Series ${i+1}:\n")
      sb.append(f"  ${"h"}%5s" + (0 until k).map(j => f"  shock${j+1}%7s").mkString + "\n")
      for (s <- 0 to h)
        sb.append(f"  $s%5d" + (0 until k).map(j =>
          f"  ${proportions(s)(i)(j)}%9.4f").mkString + "\n")
      sb.append("\n")
    }
    sb.toString
  }
}