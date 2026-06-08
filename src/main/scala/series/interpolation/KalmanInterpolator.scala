package series.interpolation

import breeze.linalg._
import series.filters.KalmanSmoother

/**
 * Kalman-based interpolation for time series with missing values.
 *
 * Missing values (NaN) are treated as unobserved states in a
 * local linear trend state space model:
 *
 *   Level:  mu_t = mu_{t-1} + nu_{t-1} + eta_t
 *   Trend:  nu_t = nu_{t-1} + zeta_t
 *   Obs:    y_t  = mu_t + eps_t
 *
 * At missing time points the observation update is skipped.
 * The RTS smoother then provides full-sample optimal interpolations.
 *
 * @param sigmaEta  level innovation std
 * @param sigmaZeta trend innovation std
 * @param sigmaEps  observation noise std
 */
class KalmanInterpolator(
  sigmaEta:  Double = 1.0,
  sigmaZeta: Double = 0.1,
  sigmaEps:  Double = 0.1
) {
  require(sigmaEta  > 0, "sigmaEta must be positive")
  require(sigmaZeta > 0, "sigmaZeta must be positive")
  require(sigmaEps  > 0, "sigmaEps must be positive")

  private val F = DenseMatrix((1.0, 1.0), (0.0, 1.0))
  private val H = DenseMatrix((1.0, 0.0))
  private val Q = DenseMatrix(
    (sigmaEta  * sigmaEta,  0.0),
    (0.0,                   sigmaZeta * sigmaZeta)
  )
  private val Rmat = DenseMatrix.create(1, 1, Array(sigmaEps * sigmaEps))

  def interpolate(y: Array[Double]): KalmanInterpResult = {
    val n   = y.length
    val obs = y.map(v => if (v.isNaN) None else Some(v))

    val x0 = DenseVector(0.0, 0.0)
    val P0 = DenseMatrix.eye[Double](2) * 1e6

    val filtMeans = new Array[DenseVector[Double]](n)
    val filtCovs  = new Array[DenseMatrix[Double]](n)
    var xCur = x0.copy
    var pCur = P0.copy

    for (t <- 0 until n) {
      val xPred = (F * xCur).copy
      val pPred = (F * pCur * F.t + Q).copy

      obs(t) match {
        case Some(yt) =>
          val innov = DenseVector(yt) - H * xPred
          val s     = H * pPred * H.t + Rmat
          val gain  = pPred * H.t * inv(s)
          xCur = (xPred + gain * innov).copy
          pCur = ((DenseMatrix.eye[Double](2) - gain * H) * pPred).copy
        case None =>
          xCur = xPred
          pCur = pPred
      }

      filtMeans(t) = xCur.copy
      filtCovs(t)  = pCur.copy
    }

    val smoother = new KalmanSmoother(F, Q)
    val smoothed = smoother.smooth(filtMeans, filtCovs)

    val interpolated = Array.tabulate(n) { t =>
      if (!y(t).isNaN) y(t)
      else smoothed.smoothedMeans(t)(0)
    }

    val variances = Array.tabulate(n)(t => smoothed.smoothedCovs(t)(0, 0))

    KalmanInterpResult(
      interpolated = interpolated,
      variances    = variances,
      original     = y,
      nMissing     = y.count(_.isNaN)
    )
  }
}

case class KalmanInterpResult(
  interpolated: Array[Double],
  variances:    Array[Double],
  original:     Array[Double],
  nMissing:     Int
) {
  val n: Int = interpolated.length
  def standardErrors: Array[Double] = variances.map(math.sqrt)
  def missingIndices: Array[Int] = original.indices.filter(i => original(i).isNaN).toArray

  override def toString: String = {
    val meanSE = if (nMissing > 0)
      missingIndices.map(i => standardErrors(i)).sum / nMissing
    else 0.0
    f"Kalman Interpolation (n=$n, missing=$nMissing)\n" +
    f"  Mean std error at missing: $meanSE%.4f\n"
  }
}
