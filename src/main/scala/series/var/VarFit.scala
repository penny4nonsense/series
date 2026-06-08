package series.`var`

import breeze.linalg._

/**
 * Results from VAR(p) estimation.
 *
 * The coefficient matrix B has shape (nCoeffs x k):
 *   Row 0:          intercept (if includeDrift)
 *   Rows 1..k:      A_1 coefficients (lag 1)
 *   Rows k+1..2k:   A_2 coefficients (lag 2)
 *   ...
 *
 * @param B            coefficient matrix (nCoeffs x k)
 * @param Sigma        residual covariance matrix (k x k)
 * @param se           standard error matrix (nCoeffs x k)
 * @param logLik       log likelihood at estimates
 * @param n            number of effective observations
 * @param p            lag order
 * @param k            number of series
 * @param Y            original data matrix
 * @param residuals    residual matrix (n-p x k)
 * @param includeDrift whether model includes a constant
 */
case class VarFit(
                   B:            DenseMatrix[Double],
                   Sigma:        DenseMatrix[Double],
                   se:           DenseMatrix[Double],
                   logLik:       Double,
                   n:            Int,
                   p:            Int,
                   k:            Int,
                   Y:            DenseMatrix[Double],
                   residuals:    DenseMatrix[Double],
                   includeDrift: Boolean
                 ) {
  private val nCoeffs = B.rows
  private val nParams = k * nCoeffs + k * (k + 1) / 2

  def aic:  Double = -2 * logLik + 2 * nParams
  def bic:  Double = -2 * logLik + math.log(n) * nParams
  def hqic: Double = -2 * logLik + 2 * math.log(math.log(n)) * nParams

  /**
   * Extracts lag coefficient matrix A_lag (k x k).
   * Row i, col j gives the effect of series j at lag `lag`
   * on series i at time t.
   */
  def lagMatrix(lag: Int): DenseMatrix[Double] = {
    require(lag >= 1 && lag <= p, s"lag must be between 1 and $p")
    val offset = if (includeDrift) 1 + (lag - 1) * k else (lag - 1) * k
    B(offset until offset + k, ::).t.copy
  }

  /**
   * Intercept vector (k x 1). Zero if no drift.
   */
  def intercept: DenseVector[Double] =
    if (includeDrift) B(0, ::).t
    else DenseVector.zeros[Double](k)

  /**
   * Companion matrix for the VAR(p) system.
   * A (kp x kp) matrix used for stability analysis and IRFs.
   */
  def companionMatrix: DenseMatrix[Double] = {
    val kp = k * p
    val C  = DenseMatrix.zeros[Double](kp, kp)

    // First k rows: lag coefficient matrices
    for (lag <- 1 to p) {
      val A = lagMatrix(lag)
      C(0 until k, (lag-1)*k until lag*k) := A
    }

    // Subdiagonal identity blocks
    if (p > 1)
      C(k until kp, 0 until k*(p-1)) := DenseMatrix.eye[Double](k*(p-1))

    C
  }

  /**
   * Checks stability — all eigenvalues of companion matrix
   * should lie inside the unit circle.
   */
  def isStable: Boolean = {
    val eigResult = eig(companionMatrix)
    val real = eigResult.eigenvalues.toArray
    val imag = eigResult.eigenvaluesComplex.toArray
    val mods = real.zip(imag).map { case (r, i) => math.sqrt(r * r + i * i) }
    mods.forall(_ < 1.0)
  }

  /**
   * Produces h-step ahead forecasts with MSE matrices.
   * Convenience wrapper over VarForecaster — the common entry point.
   */
  def forecast(h: Int): VarForecast = new VarForecaster(this).forecast(h)

  override def toString: String = {
    val sb = new StringBuilder
    sb.append(s"VAR($p) Fit — $k series, $n observations\n")
    sb.append(s"  Stable: $isStable\n")
    sb.append(f"  logLik = $logLik%.4f\n")
    sb.append(f"  AIC    = $aic%.4f\n")
    sb.append(f"  BIC    = $bic%.4f\n")
    sb.append(f"  HQIC   = $hqic%.4f\n\n")

    for (j <- 0 until k) {
      sb.append(s"  Equation $j:\n")
      sb.append(f"  ${""}%-12s  ${"coeff"}%10s  ${"std.err"}%10s\n")
      val names = (if (includeDrift) Seq("const") else Seq.empty) ++
        (for (lag <- 1 to p; i <- 0 until k) yield s"y${i+1}[${lag}]")
      for (i <- names.indices)
        sb.append(f"  ${names(i)}%-12s  ${B(i,j)}%10.4f  ${se(i,j)}%10.4f\n")
      sb.append("\n")
    }
    sb.toString
  }
}