package series.arima

import breeze.linalg._
import series.kalman._

/**
 * H-step ahead forecasts for a fitted ARIMA model.
 *
 * Forecasts are produced by running the Kalman filter to the end
 * of the observed series, then iterating the predict step forward
 * h times without an update step. Forecast variances include both
 * filter uncertainty and observation noise.
 *
 * @param arimaModel the ArimaModel specifying p, d, q
 * @param fit        the fitted ArimaFit containing parameter estimates
 */
class ArimaForecaster(arimaModel: ArimaModel, fit: ArimaFit) {

  private val kalmanModel: KalmanModel =
    arimaModel.toKalmanModel(fit.phi, fit.theta, fit.sigma2)

  def forecast(y: Array[Double], h: Int): ArimaForecast = {
    require(h > 0, "Forecast horizon must be positive")

    val diffed = difference(y, arimaModel.d)
    val obs: Seq[DenseVector[Double]] = diffed.map(v => DenseVector(v)).toIndexedSeq

    val initial    = arimaModel.initialState(fit.phi, fit.theta, fit.sigma2)
    val states     = DiffuseKalmanFilter.run(initial, kalmanModel, obs)
    val finalState = states.last

    // For pure difference models (no AR/MA), zero the state mean —
    // expected future differences are 0. For AR/MA models, use the
    // filtered state mean to propagate dynamics forward.
    val forecastX = if (arimaModel.p == 0 && arimaModel.q == 0)
      DenseVector.zeros[Double](finalState.x.length)
    else
      finalState.x

    var state = KalmanState(forecastX, finalState.Pstar)

    val pointForecasts    = Array.fill(h)(0.0)
    val forecastVariances = Array.fill(h)(0.0)

    for (i <- 0 until h) {
      val predicted = KalmanFilter.predict(state, kalmanModel)
      val yhat      = kalmanModel.H * predicted.x
      val Fmat      = kalmanModel.H * predicted.P * kalmanModel.H.t
      pointForecasts(i)    = yhat(0)
      forecastVariances(i) = Fmat(0, 0) + kalmanModel.R(0, 0)
      state = predicted
    }

    val (undiffForecasts, undiffVariances) =
      undifference(y, pointForecasts, forecastVariances, arimaModel.d)

    ArimaForecast(
      mean    = undiffForecasts,
      variance = undiffVariances,
      lower95 = undiffForecasts.zip(undiffVariances).map { case (f, v) =>
        f - 1.96 * math.sqrt(math.max(v, 0.0))
      },
      upper95 = undiffForecasts.zip(undiffVariances).map { case (f, v) =>
        f + 1.96 * math.sqrt(math.max(v, 0.0))
      }
    )
  }

  private def difference(y: Array[Double], d: Int): Array[Double] =
    if (d == 0) y
    else difference(y.zip(y.tail).map { case (a, b) => b - a }, d - 1)

  private def undifference(
                            y:         Array[Double],
                            forecasts: Array[Double],
                            variances: Array[Double],
                            d:         Int
                          ): (Array[Double], Array[Double]) = {
    if (d == 0) return (forecasts, variances)

    var level = y
    val lastValues = Array.fill(d)(0.0)
    for (i <- 0 until d) {
      lastValues(i) = level.last
      level = level.zip(level.tail).map { case (a, b) => b - a }
    }

    var f = forecasts.clone()
    var v = variances.clone()

    for (_ <- 0 until d) {
      val lastObs = lastValues(d - 1)
      f = f.scanLeft(lastObs)(_ + _).tail
      v = v.scanLeft(0.0)(_ + _).tail
    }

    (f, v)
  }
}

case class ArimaForecast(
                          mean:     Array[Double],
                          variance: Array[Double],
                          lower95:  Array[Double],
                          upper95:  Array[Double]
                        ) {
  val h: Int = mean.length

  override def toString: String = {
    val header = f"${"h"}%5s  ${"mean"}%10s  ${"lower95"}%10s  ${"upper95"}%10s"
    val rows = (0 until h).map { i =>
      f"${i+1}%5d  ${mean(i)}%10.4f  ${lower95(i)}%10.4f  ${upper95(i)}%10.4f"
    }
    (header +: rows).mkString("\n")
  }
}