package series.`var`

import breeze.linalg._

case class VarModel(p: Int, k: Int, includeDrift: Boolean = true) {
  require(p > 0, "lag order must be positive")
  require(k > 1, "VAR requires at least 2 series")

  val nCoeffs: Int = if (includeDrift) k * p + 1 else k * p

  def buildRegressors(Y: DenseMatrix[Double]): DenseMatrix[Double] = {
    val n    = Y.rows
    val nObs = n - p
    val Z    = DenseMatrix.zeros[Double](nObs, nCoeffs)

    for (t <- 0 until nObs) {
      var col = 0
      if (includeDrift) { Z(t, col) = 1.0; col += 1 }
      for (lag <- 1 to p) {
        for (j <- 0 until k) {
          Z(t, col) = Y(t + p - lag, j)
          col += 1
        }
      }
    }
    Z
  }

  def fit(Y: DenseMatrix[Double]): VarFit = {
    require(Y.cols == k, s"Expected $k series, got ${Y.cols}")
    require(Y.rows > p + nCoeffs, "Insufficient observations for lag order")

    val n      = Y.rows
    val nObs   = n - p
    val Z      = buildRegressors(Y)
    val Yresp  = Y(p until n, ::).copy

    val ZtZ    = Z.t * Z
    val ZtZInv = inv(ZtZ)
    val ZtY    = Z.t * Yresp
    val B      = ZtZInv * ZtY
    val ZB     = Z * B
    val resid  = Yresp - ZB
    val Sigma  = (resid.t * resid) / (nObs - nCoeffs).toDouble
    val ll     = logLikelihood(resid, Sigma, nObs)

    val seMatrix = DenseMatrix.zeros[Double](nCoeffs, k)
    for (j <- 0 until k) {
      val sigmaJJ = Sigma(j, j)
      for (i <- 0 until nCoeffs)
        seMatrix(i, j) = math.sqrt(sigmaJJ * ZtZInv(i, i))
    }

    VarFit(
      B            = B,
      Sigma        = Sigma,
      se           = seMatrix,
      logLik       = ll,
      n            = nObs,
      p            = p,
      k            = k,
      Y            = Y,
      residuals    = resid,
      includeDrift = includeDrift
    )
  }

  private def logLikelihood(
                             resid: DenseMatrix[Double],
                             Sigma: DenseMatrix[Double],
                             n: Int
                           ): Double = {
    val logDet = math.log(math.abs(det(Sigma)))
    val SigInv = inv(Sigma)
    // Use trace formula: tr(resid' * SigInv * resid) / n
    // More numerically stable than row-by-row
    val SigInvResid = SigInv * resid.t // k x n
    val tr = (0 until n).map { t =>
      val col = SigInvResid(::, t)
      val row = resid(t, ::).t.copy // force contiguous copy
      row.dot(col)
    }.sum
    -0.5 * (n * k * math.log(2 * math.Pi) + n * logDet + tr)
  }
}