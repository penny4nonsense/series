package series.filters

import breeze.linalg._

/**
 * Rauch-Tung-Striebel (RTS) Kalman smoother.
 *
 * Given the forward Kalman filter pass (filtered states and covariances),
 * the RTS smoother performs a backward pass to compute the
 * smoothed state estimates E[x_t | y_{1:T}].
 *
 * RTS backward recursion:
 *   G_t   = P_t|t * F' * (P_{t+1|t})^{-1}
 *   x_t|T = x_t|t + G_t * (x_{t+1|T} - x_{t+1|t})
 *   P_t|T = P_t|t + G_t * (P_{t+1|T} - P_{t+1|t}) * G_t'
 *
 * @param F  state transition matrix
 * @param Q  state noise covariance
 */
class KalmanSmoother(F: DenseMatrix[Double], Q: DenseMatrix[Double]) {

  /**
   * Runs the RTS smoother given forward filter results.
   *
   * @param filteredMeans  filtered state means (T vectors)
   * @param filteredCovs   filtered state covariances (T matrices)
   * @return SmootherResult with smoothed states and covariances
   */
  def smooth(
    filteredMeans: Array[DenseVector[Double]],
    filteredCovs:  Array[DenseMatrix[Double]]
  ): SmootherResult = {
    val T = filteredMeans.length
    require(T >= 2, "need at least 2 filtered states to smooth")

    val xSmooth = filteredMeans.map(_.copy)
    val pSmooth = filteredCovs.map(_.copy)

    for (t <- (0 until T - 1).reverse) {
      val pt    = filteredCovs(t).copy
      val pPred = (F * pt * F.t + Q).copy
      val gain  = pt * F.t * inv(pPred)

      val xPred  = F * filteredMeans(t).copy
      val innov  = xSmooth(t + 1) - xPred
      xSmooth(t) = filteredMeans(t).copy + gain * innov

      val pDiff  = pSmooth(t + 1) - pPred
      pSmooth(t) = pt + gain * pDiff * gain.t
    }

    SmootherResult(
      smoothedMeans = xSmooth.map(_.toArray),
      smoothedCovs  = pSmooth,
      filteredMeans = filteredMeans.map(_.toArray),
      filteredCovs  = filteredCovs
    )
  }
}

case class SmootherResult(
  smoothedMeans: Array[Array[Double]],
  smoothedCovs:  Array[DenseMatrix[Double]],
  filteredMeans: Array[Array[Double]],
  filteredCovs:  Array[DenseMatrix[Double]]
) {
  val T: Int        = smoothedMeans.length
  val stateDim: Int = smoothedMeans.head.length

  def varianceReduction: Array[Double] =
    Array.tabulate(T) { t =>
      val filtVar   = trace(filteredCovs(t))
      val smoothVar = trace(smoothedCovs(t))
      if (filtVar > 1e-10) (filtVar - smoothVar) / filtVar else 0.0
    }

  override def toString: String = {
    val avgReduction = varianceReduction.sum / T
    f"RTS Smoother Result (T=$T, stateDim=$stateDim)\n" +
    f"  Average variance reduction: ${avgReduction * 100}%.2f%%\n"
  }
}
