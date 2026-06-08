package series.vecm

import breeze.linalg._

/**
 * Impulse response functions for VECM.
 *
 * For cointegrated systems IRFs have special properties:
 *
 *   Transitory shocks — IRF converges to zero at long horizons
 *   Permanent shocks  — IRF converges to a nonzero constant
 *                       (the long-run effect)
 *
 * Persistence profiles (Pesaran & Shin 1996) measure how long
 * it takes for the cointegrating relationships to return to
 * equilibrium after a system-wide shock.
 *
 * @param fit VecmFit with parameter estimates
 */
class VecmIRF(fit: VecmFit) {

  private val k       = fit.k
  private val varMats = fit.toVarMatrices

  /**
   * Computes MA coefficient matrices Phi_0..Phi_h.
   */
  private def maCoefficients(h: Int): Array[DenseMatrix[Double]] = {
    val phis = Array.fill(h + 1)(DenseMatrix.zeros[Double](k, k))
    phis(0) = DenseMatrix.eye[Double](k)
    for (s <- 1 to h)
      for (lag <- 1 to math.min(s, fit.p))
        phis(s) += varMats(lag - 1) * phis(s - lag)
    phis
  }

  /**
   * Orthogonalized IRFs via Cholesky decomposition of Sigma.
   *
   * @param h forecast horizon
   * @return IRF matrices (h+1 x k x k)
   */
  def orthogonalIRF(h: Int): Array[DenseMatrix[Double]] = {
    require(h > 0, "horizon must be positive")
    val P    = cholesky(fit.Sigma)
    val phis = maCoefficients(h)
    phis.map(_ * P)
  }

  /**
   * Generalized IRFs (Pesaran & Shin 1998).
   * Ordering-invariant.
   *
   * @param h forecast horizon
   * @return IRF matrices (h+1 x k x k)
   */
  def generalizedIRF(h: Int): Array[DenseMatrix[Double]] = {
    require(h > 0, "horizon must be positive")
    val phis  = maCoefficients(h)
    val Sigma = fit.Sigma

    phis.map { phi =>
      DenseMatrix.tabulate(k, k) { (i, j) =>
        val ei    = DenseVector.zeros[Double](k); ei(i) = 1.0
        val ej    = DenseVector.zeros[Double](k); ej(j) = 1.0
        val sigJJ = Sigma(j, j)
        (ei.t * phi * Sigma * ej) / math.sqrt(sigJJ)
      }
    }
  }

  /**
   * Long-run (cumulative) IRF matrix.
   * Converges to C(1) * P for orthogonal identification
   * where C(1) = (I - A_1 - ... - A_p)^{-1}
   *
   * For permanent shocks this is nonzero.
   * For transitory shocks this is zero.
   *
   * @param h horizon to accumulate to
   * @return cumulative IRF matrix (k x k) at horizon h
   */
  def cumulativeIRF(h: Int): DenseMatrix[Double] = {
    val P    = cholesky(fit.Sigma)
    val phis = maCoefficients(h)
    phis.map(_ * P).reduce(_ + _)
  }

  /**
   * Persistence profiles of cointegrating relationships.
   *
   * Measures how long the cointegrating relationship z_t = beta' Y_t
   * takes to return to equilibrium after a system-wide shock.
   *
   * For a correctly specified VECM, persistence profiles should
   * converge to zero — the cointegrating relationship is stationary.
   *
   * Pesaran & Shin (1996) definition:
   *   h(s) = beta' * Phi_s * Sigma * Phi_s' * beta /
   *           beta' * Sigma * beta
   *
   * @param h  maximum horizon
   * @return array of persistence profile values (h+1)
   *         one per cointegrating vector
   */
  def persistenceProfiles(h: Int): Array[Array[Double]] = {
    require(h > 0, "horizon must be positive")
    val phis  = maCoefficients(h)
    val Sigma = fit.Sigma
    val beta  = fit.beta

    (0 until fit.r).map { j =>
      val betaJ  = beta(::, j)
      val denom  = betaJ.t * Sigma * betaJ

      (0 to h).map { s =>
        val numer = betaJ.t * phis(s) * Sigma * phis(s).t * betaJ
        numer / math.max(denom, 1e-10)
      }.toArray
    }.toArray
  }

  /**
   * Forecast error variance decomposition.
   *
   * @param h forecast horizon
   * @return FEVD array (h+1 x k x k) proportions
   */
  def fevd(h: Int): Array[Array[Array[Double]]] = {
    val sirfs = orthogonalIRF(h)
    val contributions = Array.tabulate(h+1, k, k) { (s, i, j) =>
      sirfs(s)(i, j) * sirfs(s)(i, j)
    }
    val cumulative = Array.tabulate(h+1, k, k) { (s, i, j) =>
      (0 to s).map(t => contributions(t)(i)(j)).sum
    }
    Array.tabulate(h+1, k, k) { (s, i, j) =>
      val total = (0 until k).map(jj => cumulative(s)(i)(jj)).sum
      if (total > 0) cumulative(s)(i)(j) / total else 0.0
    }
  }
}