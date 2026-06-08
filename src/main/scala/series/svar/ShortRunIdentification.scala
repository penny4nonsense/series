package series.svar

import breeze.linalg._

/**
 * Short-run (impact) restriction identification for SVAR.
 *
 * Two approaches:
 *
 * 1. Cholesky (recursive) identification — B_0 is lower triangular.
 *    The impact matrix B0inv = P (Cholesky factor of Sigma).
 *    Simple, exact, but ordering-dependent.
 *    Sims (1980).
 *
 * 2. General zero restrictions — arbitrary zero restrictions on B_0.
 *    Solved by a nonlinear equation system.
 *    Requires exactly k(k-1)/2 restrictions for exact identification.
 *
 * Short-run restrictions constrain the CONTEMPORANEOUS effect of
 * structural shocks on variables.
 */
object ShortRunIdentification {

  /**
   * Cholesky identification.
   *
   * Imposes a recursive causal ordering — variable j is not
   * contemporaneously affected by variable i if i > j.
   *
   * The impact matrix B0inv = P where P is the lower Cholesky
   * factor of Sigma, i.e. P * P' = Sigma.
   *
   * @param model  SvarModel
   * @param h      IRF horizon
   * @param shockNames optional names for structural shocks
   * @return SvarResult with structural IRFs
   */
  def cholesky(
                model:      SvarModel,
                h:          Int,
                shockNames: Option[Array[String]] = None
              ): SvarResult = {
    val B0inv = model.choleskyFactor
    val sirfs = model.structuralIRF(B0inv, h)
    val shocks = model.structuralShocks(B0inv)

    SvarResult(
      B0inv        = B0inv,
      sirfs        = sirfs,
      shocks       = shocks,
      fit          = model.fit,
      h            = h,
      identification = "Cholesky (short-run)",
      shockNames   = shockNames.getOrElse(
        (0 until model.k).map(j => s"shock${j+1}").toArray
      )
    )
  }

  /**
   * General zero restrictions on impact matrix B_0.
   *
   * Restrictions are specified as a list of (row, col) pairs
   * where B_0(row, col) = 0, meaning structural shock `col`
   * has no contemporaneous effect on variable `row`.
   *
   * Note: B_0 is the inverse of the impact matrix B0inv.
   * A zero in B_0 restricts which shocks affect which variables
   * on impact.
   *
   * Solved iteratively using the scoring algorithm of
   * Amisano & Giannini (1997).
   *
   * @param model        SvarModel
   * @param restrictions list of (row, col) zero restrictions on B_0
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

    // Initialize B_0 from Cholesky
    val B0init = inv(model.choleskyFactor)

    // Scoring algorithm — iteratively update B_0
    var B0 = B0init.copy
    var iter = 0
    var converged = false

    while (iter < maxIter && !converged) {
      val B0inv = inv(B0)
      val Sigma = model.Sigma
      val SigInv = inv(Sigma)

      // Gradient of log likelihood wrt B_0
      // dL/dB_0 = (B_0^{-T} - Sigma^{-1} * B_0^{-T} * ... )
      // Simplified scoring update
      val grad = B0inv.t - Sigma * B0.t

      // Zero out restricted elements in gradient
      for ((row, col) <- restrictions)
        grad(row, col) = 0.0

      val B0new = B0 + grad * 0.01

      // Re-enforce restrictions
      for ((row, col) <- restrictions)
        B0new(row, col) = 0.0

      val diff = norm((B0new - B0).toDenseVector)
      B0 = B0new
      iter += 1
      if (diff < tol) converged = true
    }

    val B0inv = inv(B0)
    val sirfs  = model.structuralIRF(B0inv, h)
    val shocks = model.structuralShocks(B0inv)

    SvarResult(
      B0inv          = B0inv,
      sirfs          = sirfs,
      shocks         = shocks,
      fit            = model.fit,
      h              = h,
      identification = "Zero restrictions (short-run)",
      shockNames     = shockNames.getOrElse(
        (0 until k).map(j => s"shock${j+1}").toArray
      )
    )
  }
}