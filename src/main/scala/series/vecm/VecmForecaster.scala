package series.vecm

import breeze.linalg._

class VecmForecaster(fit: VecmFit) {

  def forecast(h: Int): VecmForecast = {
    require(h > 0, "forecast horizon must be positive")

    val k       = fit.k
    val p       = fit.p
    val n       = fit.Y.rows
    val varMats = fit.toVarMatrices

    val history = scala.collection.mutable.ArrayBuffer[DenseVector[Double]]()
    for (t <- n - p until n)
      history += fit.Y(t, ::).t.copy

    val forecasts = DenseMatrix.zeros[Double](h, k)

    for (i <- 0 until h) {
      var yhat = if (fit.includeDrift) fit.intercept.copy
                 else DenseVector.zeros[Double](k)
      for (lag <- 0 until p) {
        val ylag = history(history.length - 1 - lag).copy
        yhat += varMats(lag) * ylag
      }
      forecasts(i, ::) := yhat.t
      history += yhat
    }

    val maCoeffs = computeMACoefficients(varMats, h)
    val mse = Array.fill(h)(DenseMatrix.zeros[Double](k, k))
    for (s <- 0 until h) {
      for (t <- 0 to s) {
        val phi    = maCoeffs(t).copy
        val phiSig = phi * fit.Sigma.copy
        mse(s) += phiSig * phi.t
      }
    }

    val cointRelationships = DenseMatrix.tabulate(h, fit.r) { (t, j) =>
      fit.beta(::, j).t * forecasts(t, ::).t
    }

    VecmForecast(
      forecasts          = forecasts,
      mse                = mse,
      cointRelationships = cointRelationships,
      h                  = h,
      k                  = k,
      r                  = fit.r
    )
  }

  private def computeMACoefficients(
    varMats: Array[DenseMatrix[Double]],
    h:       Int
  ): Array[DenseMatrix[Double]] = {
    val k    = fit.k
    val phis = Array.fill(h + 1)(DenseMatrix.zeros[Double](k, k))
    phis(0) = DenseMatrix.eye[Double](k)
    for (s <- 1 to h) {
      for (lag <- 1 to math.min(s, fit.p)) {
        val a    = varMats(lag - 1).copy
        val prev = phis(s - lag).copy
        phis(s) += a * prev
      }
    }
    phis
  }
}

case class VecmForecast(
  forecasts:          DenseMatrix[Double],
  mse:                Array[DenseMatrix[Double]],
  cointRelationships: DenseMatrix[Double],
  h:                  Int,
  k:                  Int,
  r:                  Int
) {
  def intervalHalfWidths: DenseMatrix[Double] =
    DenseMatrix.tabulate(h, k) { (i, j) =>
      1.96 * math.sqrt(mse(i)(j, j))
    }

  override def toString: String = {
    val sb = new StringBuilder
    sb.append("VECM Forecasts:\n")
    sb.append(s"  h" + (0 until k).map(j => s"  y${j+1}").mkString + "\n")
    for (i <- 0 until h)
      sb.append(f"  ${i+1}%5d" + (0 until k).map(j =>
        f"  ${forecasts(i,j)}%10.4f").mkString + "\n")
    sb.append("\n  Cointegrating relationships:\n")
    sb.append(s"  h" + (0 until r).map(j => s"  z${j+1}").mkString + "\n")
    for (i <- 0 until h)
      sb.append(f"  ${i+1}%5d" + (0 until r).map(j =>
        f"  ${cointRelationships(i,j)}%10.4f").mkString + "\n")
    sb.toString
  }
}
