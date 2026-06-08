package series.svar

import breeze.linalg._
import breeze.linalg.svd.SVD

/**
 * Sign restriction identification for SVAR.
 *
 * Uhlig (2005) — shocks are identified by restricting the signs
 * of impulse responses over some horizon rather than imposing
 * zero restrictions.
 *
 * The model is set-identified — there is a whole set of impact
 * matrices consistent with the sign restrictions. The algorithm
 * draws random orthogonal matrices from the Haar measure and
 * keeps those satisfying the restrictions.
 *
 * Restrictions are specified as:
 *   SignRestriction(series, shock, horizon, sign)
 * meaning the IRF of `series` to `shock` at `horizon` must
 * have the given sign (+1 or -1).
 *
 * Returns a distribution of admissible structural IRFs,
 * summarized by median and credible bands.
 */
object SignRestrictions {

  /**
   * A single sign restriction.
   *
   * @param series  index of responding series (0-based)
   * @param shock   index of structural shock (0-based)
   * @param horizon horizon at which restriction applies (0 = impact)
   * @param sign    required sign (+1 or -1)
   */
  case class SignRestriction(
                              series:  Int,
                              shock:   Int,
                              horizon: Int,
                              sign:    Int
                            ) {
    require(sign == 1 || sign == -1, "sign must be +1 or -1")
  }

  /**
   * Draws a random orthogonal matrix from the Haar measure
   * using QR decomposition of a random normal matrix.
   *
   * @param k    dimension
   * @param rng  random number generator
   * @return random orthogonal matrix (k x k)
   */
  private def randomOrthogonal(k: Int, rng: scala.util.Random): DenseMatrix[Double] = {
    val X = DenseMatrix.tabulate(k, k)((_, _) => rng.nextGaussian())
    val SVD(u, _, _) = svd(X)
    u
  }

  /**
   * Checks whether a candidate impact matrix satisfies all
   * sign restrictions.
   *
   * @param sirfs        structural IRFs
   * @param restrictions list of sign restrictions
   * @return true if all restrictions are satisfied
   */
  private def satisfiesRestrictions(
                                     sirfs:        Array[DenseMatrix[Double]],
                                     restrictions: Seq[SignRestriction]
                                   ): Boolean =
    restrictions.forall { r =>
      val irf = sirfs(r.horizon)(r.series, r.shock)
      if (r.sign == 1) irf >= 0.0 else irf <= 0.0
    }

  /**
   * Identifies structural shocks via sign restrictions using
   * the Uhlig (2005) rejection algorithm.
   *
   * Draws nDraws random orthogonal matrices, computes candidate
   * structural IRFs, and keeps those satisfying all restrictions.
   *
   * @param model        SvarModel
   * @param restrictions list of SignRestriction objects
   * @param h            IRF horizon
   * @param nDraws       number of random draws to attempt
   * @param seed         random seed for reproducibility
   * @return SignRestrictionResult with distribution of admissible IRFs
   */
  def identify(
                model:        SvarModel,
                restrictions: Seq[SignRestriction],
                h:            Int,
                nDraws:       Int = 10000,
                seed:         Long = 42,
                shockNames:   Option[Array[String]] = None
              ): SignRestrictionResult = {
    require(restrictions.nonEmpty, "Must provide at least one restriction")
    require(nDraws > 0, "nDraws must be positive")

    val k   = model.k
    val rng = new scala.util.Random(seed)
    val P   = model.choleskyFactor

    val admissibleIRFs   = scala.collection.mutable.ArrayBuffer[Array[DenseMatrix[Double]]]()
    val admissibleB0invs = scala.collection.mutable.ArrayBuffer[DenseMatrix[Double]]()

    var draws    = 0
    var accepted = 0

    while (draws < nDraws) {
      // Draw random orthogonal matrix Q
      val Q = randomOrthogonal(k, rng)

      // Candidate impact matrix: B0inv = P * Q
      val B0inv = P * Q

      // Compute structural IRFs
      val sirfs = model.structuralIRF(B0inv, h)

      // Check sign restrictions
      if (satisfiesRestrictions(sirfs, restrictions)) {
        admissibleIRFs   += sirfs
        admissibleB0invs += B0inv
        accepted += 1
      }

      draws += 1
    }

    require(accepted > 0,
      s"No draws satisfied the sign restrictions after $nDraws attempts. " +
        "Check that restrictions are mutually consistent.")

    SignRestrictionResult(
      admissibleIRFs   = admissibleIRFs.toArray,
      admissibleB0invs = admissibleB0invs.toArray,
      nDraws           = nDraws,
      nAccepted        = accepted,
      h                = h,
      k                = k,
      restrictions     = restrictions,
      shockNames       = shockNames.getOrElse(
        (0 until k).map(j => s"shock${j+1}").toArray
      )
    )
  }
}