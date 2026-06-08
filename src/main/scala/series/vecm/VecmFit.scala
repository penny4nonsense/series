package series.vecm

import breeze.linalg._

/**
 * Results from VECM estimation.
 *
 * @param alpha        adjustment speed matrix (k x r)
 * @param beta         cointegrating vector matrix (k x r)
 * @param Pi           long-run impact matrix alpha * beta' (k x k)
 * @param gammas       short-run dynamic matrices (p-1 matrices of k x k)
 * @param intercept    intercept vector (k)
 * @param Sigma        residual covariance (k x k)
 * @param logLik       log likelihood
 * @param eigenvalues  Johansen eigenvalues (r largest)
 * @param n            effective observations
 * @param k            number of series
 * @param p            VAR lag order
 * @param r            cointegrating rank
 * @param Y            original data
 * @param residuals    VECM residuals (n-p x k)
 * @param includeDrift whether model includes constant
 */
case class VecmFit(
                    alpha:        DenseMatrix[Double],
                    beta:         DenseMatrix[Double],
                    Pi:           DenseMatrix[Double],
                    gammas:       Array[DenseMatrix[Double]],
                    intercept:    DenseVector[Double],
                    Sigma:        DenseMatrix[Double],
                    logLik:       Double,
                    eigenvalues:  Array[Double],
                    n:            Int,
                    k:            Int,
                    p:            Int,
                    r:            Int,
                    Y:            DenseMatrix[Double],
                    residuals:    DenseMatrix[Double],
                    includeDrift: Boolean
                  ) {
  private val nParams = r * k + (k - r) * r +
    gammas.length * k * k +
    (if (includeDrift) k else 0) +
    k * (k + 1) / 2

  def aic:  Double = -2 * logLik + 2 * nParams
  def bic:  Double = -2 * logLik + math.log(n) * nParams
  def hqic: Double = -2 * logLik + 2 * math.log(math.log(n)) * nParams

  /**
   * Cointegrating relationship j evaluated at time t.
   * z_{t,j} = beta_j' * Y_t
   * Should be stationary under correct specification.
   *
   * @param j cointegrating vector index (0-based)
   * @return cointegrating relationship time series
   */
  def cointegratingRelationship(j: Int): Array[Double] = {
    require(j >= 0 && j < r, s"j must be between 0 and ${r-1}")
    val betaJ = beta(::, j)
    (0 until Y.rows).map(t => betaJ.t * Y(t, ::).t).toArray
  }

  /**
   * All cointegrating relationships as a matrix (n x r).
   */
  def cointegratingRelationships: DenseMatrix[Double] =
    DenseMatrix.tabulate(Y.rows, r)((t, j) =>
      beta(::, j).t * Y(t, ::).t
    )

  /**
   * Speed of adjustment for series i to cointegrating vector j.
   * Negative values indicate mean-reverting behavior.
   */
  def adjustmentSpeed(series: Int, cointVec: Int): Double =
    alpha(series, cointVec)

  /**
   * Converts VECM back to VAR(p) representation.
   * Useful for forecasting and IRF computation.
   *
   * The companion VAR(p) has coefficient matrices:
   *   A_1 = I + Pi + Gamma_1
   *   A_j = Gamma_j - Gamma_{j-1}  for j = 2..p-1
   *   A_p = -Gamma_{p-1}
   */
  def toVarMatrices: Array[DenseMatrix[Double]] = {
    val Ik = DenseMatrix.eye[Double](k)
    val nLags = gammas.length

    if (nLags == 0) {
      Array(Ik + Pi)
    } else {
      val A = Array.fill(p)(DenseMatrix.zeros[Double](k, k))
      A(0) = Ik + Pi + gammas(0)
      for (j <- 1 until nLags)
        A(j) = gammas(j) - gammas(j-1)
      A(nLags) = -gammas(nLags - 1)
      A
    }
  }

  /**
   * Companion matrix for stability analysis.
   */
  def companionMatrix: DenseMatrix[Double] = {
    val varMats = toVarMatrices
    val kp      = k * p
    val C       = DenseMatrix.zeros[Double](kp, kp)

    for (lag <- 0 until p)
      C(0 until k, lag*k until (lag+1)*k) := varMats(lag)

    if (p > 1)
      C(k until kp, 0 until k*(p-1)) := DenseMatrix.eye[Double](k*(p-1))

    C
  }

  /**
   * Checks stability — exactly r eigenvalues of companion matrix
   * should be on the unit circle (cointegration), rest inside.
   */
  def isCointegrated: Boolean = {
    val eigs = eig(companionMatrix)
    val mods = eigs.eigenvalues.toArray
      .zip(eigs.eigenvaluesComplex.toArray)
      .map { case (re, im) => math.sqrt(re*re + im*im) }
    val unitRoot  = mods.count(m => math.abs(m - 1.0) < 0.1)
    val insideUnit = mods.count(_ < 0.999)
    unitRoot == r && insideUnit == k*p - r
  }

  /**
   * Produces h-step ahead level forecasts with MSE matrices and
   * forecasted cointegrating relationships. Convenience wrapper over
   * VecmForecaster — the common entry point.
   */
  def forecast(h: Int): VecmForecast = new VecmForecaster(this).forecast(h)

  override def toString: String = {
    val sb = new StringBuilder
    sb.append(s"VECM($p, r=$r) — $k series, $n observations\n\n")

    sb.append("  Cointegrating vectors beta' (normalized):\n")
    for (j <- 0 until r) {
      sb.append(s"  beta${j+1}: ")
      for (i <- 0 until k)
        sb.append(f"${beta(i,j)}%10.4f")
      sb.append("\n")
    }
    sb.append("\n")

    sb.append("  Adjustment speeds alpha:\n")
    for (i <- 0 until k) {
      sb.append(s"  y${i+1}:    ")
      for (j <- 0 until r)
        sb.append(f"${alpha(i,j)}%10.4f")
      sb.append("\n")
    }
    sb.append("\n")

    sb.append("  Johansen eigenvalues:\n")
    eigenvalues.zipWithIndex.foreach { case (l, j) =>
      sb.append(f"  lambda${j+1} = $l%.6f\n")
    }
    sb.append("\n")

    sb.append(f"  logLik = $logLik%.4f\n")
    sb.append(f"  AIC    = $aic%.4f\n")
    sb.append(f"  BIC    = $bic%.4f\n")
    sb.append(f"  HQIC   = $hqic%.4f\n")
    sb.append(f"  Cointegrated: $isCointegrated\n")
    sb.toString
  }
}