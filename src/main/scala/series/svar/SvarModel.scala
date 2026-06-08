package series.svar

import breeze.linalg._
import series.`var`.{VarFit, ImpulseResponse}

/**
 * Structural VAR model.
 *
 * The structural form is:
 *   B_0 * Y_t = c + B_1 * Y_{t-1} + ... + B_p * Y_{t-p} + epsilon_t
 *
 * where epsilon_t ~ N(0, I) are orthogonal structural shocks.
 *
 * The reduced form VAR gives:
 *   Y_t = A_1 * Y_{t-1} + ... + A_p * Y_{t-p} + u_t
 *
 * where u_t = B_0^{-1} * epsilon_t and Sigma = B_0^{-1} (B_0^{-1})'
 *
 * Identification requires k(k-1)/2 restrictions on B_0 to recover
 * the structural shocks from the reduced form residuals.
 *
 * @param fit     fitted reduced form VarFit
 */
class SvarModel(val fit: VarFit) {

  val k: Int = fit.k
  val p: Int = fit.p

  /**
   * Reduced form residuals (n-p x k).
   */
  val reducedFormResiduals: DenseMatrix[Double] = fit.residuals

  /**
   * Reduced form covariance matrix Sigma.
   */
  val Sigma: DenseMatrix[Double] = fit.Sigma

  /**
   * Cholesky factor of Sigma — lower triangular P such that P*P' = Sigma.
   * Used as starting point for short-run identification.
   */
  val choleskyFactor: DenseMatrix[Double] = cholesky(Sigma)

  /**
   * Computes structural IRFs given impact matrix B0inv = B_0^{-1}.
   *
   * Structural IRF at horizon s:
   *   Theta_s = Phi_s * B0inv
   *
   * where Phi_s are the reduced form MA coefficients.
   *
   * @param B0inv impact matrix (k x k)
   * @param h     forecast horizon
   * @return structural IRF matrices (h+1 x k x k)
   *         result(s)(i,j) = response of series i at horizon s
   *         to structural shock j
   */
  def structuralIRF(
                     B0inv: DenseMatrix[Double],
                     h:     Int
                   ): Array[DenseMatrix[Double]] = {
    val irf  = new ImpulseResponse(fit)
    val phis = irf.maCoefficients(h)
    phis.map(_ * B0inv)
  }

  /**
   * Cumulative structural IRFs — sum of responses from 0 to h.
   * Used for long-run restriction identification.
   *
   * @param B0inv impact matrix
   * @param h     horizon to accumulate to
   * @return cumulative IRF matrix (k x k) at horizon h
   */
  def cumulativeIRF(
                     B0inv: DenseMatrix[Double],
                     h:     Int
                   ): DenseMatrix[Double] = {
    val sirfs = structuralIRF(B0inv, h)
    sirfs.reduce(_ + _)
  }

  /**
   * Long-run multiplier matrix — the infinite sum of MA coefficients.
   * For stable VAR: C(1) = (I - A_1 - ... - A_p)^{-1}
   * This is the key object for Blanchard-Quah identification.
   */
  def longRunMultiplier: DenseMatrix[Double] = {
    val Ik = DenseMatrix.eye[Double](k)
    var sum = Ik.copy
    for (lag <- 1 to p)
      sum -= fit.lagMatrix(lag)
    inv(sum)
  }

  /**
   * Extracts structural shocks given impact matrix B0inv.
   * epsilon_t = B_0 * u_t = B0inv^{-1} * u_t
   *
   * @param B0inv impact matrix
   * @return structural shock matrix (n-p x k)
   */
  def structuralShocks(B0inv: DenseMatrix[Double]): DenseMatrix[Double] = {
    val B0 = inv(B0inv)
    reducedFormResiduals * B0.t
  }

  /**
   * Verifies that B0inv is consistent with Sigma.
   * Should satisfy: B0inv * B0inv' ≈ Sigma
   *
   * @param B0inv impact matrix
   * @param tol   tolerance
   */
  def verifySigma(
                   B0inv: DenseMatrix[Double],
                   tol:   Double = 1e-6
                 ): Boolean = {
    val diff = B0inv * B0inv.t - Sigma
    norm(diff.toDenseVector) < tol
  }
}