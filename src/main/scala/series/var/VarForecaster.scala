package series.`var`

import breeze.linalg._

/**
 * H-step ahead forecasts for a fitted VAR model.
 *
 * Everything needed for forecasting is carried by the VarFit, so the
 * forecaster takes only the fit. The common path is `fit.forecast(h)`.
 *
 * @param fit VarFit with parameter estimates
 */
class VarForecaster(fit: VarFit) {

  def forecast(h: Int): VarForecast = {
    require(h > 0, "forecast horizon must be positive")

    val k = fit.k
    val p = fit.p
    val n = fit.Y.rows

    // Initialize with last p observations — copy to avoid view issues
    val history = scala.collection.mutable.ArrayBuffer[DenseVector[Double]]()
    for (t <- 0 until p)
      history += fit.Y(n - p + t, ::).t.copy

    // Iterate VAR recursion forward
    val forecasts = DenseMatrix.zeros[Double](h, k)
    for (i <- 0 until h) {
      var yhat = fit.intercept.copy
      for (lag <- 1 to p) {
        val A    = fit.lagMatrix(lag)       // already returns .copy
        val ylag = history(history.length - lag).copy
        val Aylag = A * ylag
        yhat += Aylag
      }
      forecasts(i, ::) := yhat.t
      history += yhat
    }

    // Forecast MSE: sum_{s=0}^{h-1} Phi_s * Sigma * Phi_s'
    val irf      = new ImpulseResponse(fit)
    val allPhis  = irf.maCoefficients(h - 1)
    val Sigma    = fit.Sigma.copy

    val msMatrices = Array.fill(h)(DenseMatrix.zeros[Double](k, k))
    for (s <- 0 until h) {
      var mse = DenseMatrix.zeros[Double](k, k)
      for (t <- 0 to s) {
        val phi    = allPhis(t).copy
        val phiSig = phi * Sigma
        val contrib = phiSig * phi.t
        mse += contrib
      }
      msMatrices(s) = mse
    }

    VarForecast(forecasts = forecasts, mse = msMatrices, h = h, k = k)
  }
}

case class VarForecast(
  forecasts: DenseMatrix[Double],
  mse:       Array[DenseMatrix[Double]],
  h:         Int,
  k:         Int
) {
  def intervalHalfWidths: DenseMatrix[Double] =
    DenseMatrix.tabulate(h, k) { (i, j) =>
      1.96 * math.sqrt(mse(i)(j, j))
    }

  override def toString: String = {
    val sb = new StringBuilder
    sb.append("VAR Forecasts:\n")
    sb.append(f"  ${"h"}%5s" + (0 until k).map(j => s"  y${j+1}").mkString + "\n")
    for (i <- 0 until h)
      sb.append(f"  ${i+1}%5d" + (0 until k).map(j =>
        f"  ${forecasts(i,j)}%10.4f").mkString + "\n")
    sb.toString
  }
}
