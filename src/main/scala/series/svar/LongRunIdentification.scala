package series.svar

import breeze.linalg._

/**
 * Long-run restriction identification for SVAR.
 *
 * Blanchard and Quah (1989) identification.
 *
 * Some structural shocks are restricted to have zero cumulative
 * effect on some variables — i.e. their long-run multiplier is zero.
 *
 * Classic example (Blanchard-Quah 1989):
 *   - Demand shocks have no long-run effect on output
 *   - Supply shocks can have permanent effects on output
 *
 * The long-run multiplier matrix is:
 *   C(1) = (I - A_1 - ... - A_p)^{-1}
 *
 * The structural long-run multiplier is:
 *   D = C(1) * B0inv
 *
 * Restrictions are placed on D — typically that D is lower triangular,
 * meaning shocks ordered later have no long-run effect on variables
 * ordered earlier.
 *
 * Identification:
 *   D = C(1) * B0inv must be lower triangular
 *   => B0inv = C(1)^{-1} * D
 *   => Cholesky decompose C(1) * Sigma * C(1)' to get D
 *   => B0inv = C(1)^{-1} * D
 */
object LongRunIdentification {

  /**
   * Blanchard-Quah identification with lower triangular long-run
   * multiplier matrix.
   *
   * The ordering of variables determines which shocks have long-run
   * effects on which variables. Variable j is not permanently affected
   * by shock i if i > j.
   *
   * @param model      SvarModel
   * @param h          IRF horizon
   * @param shockNames optional shock names
   * @return SvarResult with structural IRFs
   */
  def blanchardQuah(
                     model:      SvarModel,
                     h:          Int,
                     shockNames: Option[Array[String]] = None
                   ): SvarResult = {
    val C1    = model.longRunMultiplier
    val Sigma = model.Sigma

    // C(1) * Sigma * C(1)' — covariance of long-run reduced form multiplier
    val longRunSigma = C1 * Sigma * C1.t

    // Cholesky decompose to get lower triangular D
    val D = cholesky(longRunSigma)

    // Recover impact matrix B0inv = C(1)^{-1} * D
    val B0inv = inv(C1) * D

    // Verify Sigma consistency
    require(model.verifySigma(B0inv, tol=1e-4),
      "Long-run identification failed sigma consistency check")

    val sirfs  = model.structuralIRF(B0inv, h)
    val shocks = model.structuralShocks(B0inv)

    SvarResult(
      B0inv          = B0inv,
      sirfs          = sirfs,
      shocks         = shocks,
      fit            = model.fit,
      h              = h,
      identification = "Blanchard-Quah (long-run)",
      shockNames     = shockNames.getOrElse(
        (0 until model.k).map(j => s"shock${j+1}").toArray
      )
    )
  }

  /**
   * General long-run zero restrictions.
   *
   * Allows arbitrary zero restrictions on the long-run multiplier
   * matrix D = C(1) * B0inv, not just lower triangularity.
   *
   * Restrictions: D(row, col) = 0 means shock `col` has no
   * long-run effect on variable `row`.
   *
   * Requires exactly k(k-1)/2 restrictions for exact identification.
   *
   * Solved by the algorithm of Vlaar (2004) — iterative refinement
   * of the impact matrix subject to long-run constraints.
   *
   * @param model        SvarModel
   * @param restrictions list of (row, col) zero restrictions on D
   * @param h            IRF horizon
   * @param maxIter      maximum iterations
   * @param tol          convergence tolerance
   * @param shockNames   optional shock names
   * @return SvarResult with structural IRFs
   */
  def zeroRestrictions(
                        model:        SvarModel,
                        restrictions: Seq[(Int, Int)],
                        h:            Int,
                        maxIter:      Int = 1000,
                        tol:          Double = 1e-8,
                        shockNames:   Option[Array[String]] = None
                      ): SvarResult = {
    val k     = model.k
    val nReqs = k * (k - 1) / 2
    require(restrictions.length == nReqs,
      s"Exact identification requires $nReqs restrictions, got ${restrictions.length}")

    val C1    = model.longRunMultiplier
    val Sigma = model.Sigma

    // Start from Blanchard-Quah solution
    val longRunSigma = C1 * Sigma * C1.t
    var D = cholesky(longRunSigma)

    var iter = 0
    var converged = false

    while (iter < maxIter && !converged) {
      // Enforce restrictions on D
      for ((row, col) <- restrictions)
        D(row, col) = 0.0

      // Reorthogonalize — project onto constraint set
      val B0inv = inv(C1) * D
      val Dnew  = C1 * B0inv

      val diff = norm((Dnew - D).toDenseVector)
      D = Dnew
      iter += 1
      if (diff < tol) converged = true
    }

    // Enforce restrictions one final time
    for ((row, col) <- restrictions)
      D(row, col) = 0.0

    val B0inv = inv(C1) * D
    val sirfs  = model.structuralIRF(B0inv, h)
    val shocks = model.structuralShocks(B0inv)

    SvarResult(
      B0inv          = B0inv,
      sirfs          = sirfs,
      shocks         = shocks,
      fit            = model.fit,
      h              = h,
      identification = "Zero restrictions (long-run)",
      shockNames     = shockNames.getOrElse(
        (0 until k).map(j => s"shock${j+1}").toArray
      )
    )
  }
}