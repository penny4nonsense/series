package series.arima

import breeze.linalg._
import series.kalman._

/**
 * H-step ahead forecasts for a fitted SARIMA model.
 *
 * Forecasts are produced by running the diffuse Kalman filter to the
 * end of the observed series, then iterating the predict step forward
 * h times without update. Undifferencing reverses both nonseasonal
 * and seasonal differencing in the correct order.
 *
 * @param sarimaModel the SarimaModel specifying orders and seasonal period
 * @param fit         the fitted SarimaFit containing parameter estimates
 */
class SarimaForecaster(sarimaModel: SarimaModel, fit: SarimaFit) {

  private val kalmanModel: KalmanModel =
    sarimaModel.toKalmanModel(fit.phi, fit.phiS, fit.theta, fit.thetaS, fit.sigma2)

  /**
   * Produces h-step ahead point forecasts and forecast variances.
   *
   * @param y observed series (undifferenced, undeseasonalized)
   * @param h forecast horizon
   * @return SarimaForecast with point forecasts and prediction intervals
   */
  def forecast(y: Array[Double], h: Int): SarimaForecast = {
    require(h > 0, "Forecast horizon must be positive")
    require(y.length > sarimaModel.s * sarimaModel.D + sarimaModel.d,
      "Series too short for specified differencing orders")

    val diffed = difference(y)
    val obs: Seq[DenseVector[Double]] = diffed.map(v => DenseVector(v)).toIndexedSeq

    val initial = sarimaModel.initialState(
      fit.phi, fit.phiS, fit.theta, fit.thetaS, fit.sigma2
    )
    val states     = DiffuseKalmanFilter.run(initial, kalmanModel, obs)
    val finalState = states.last

    var state = KalmanState(finalState.x, finalState.Pstar)

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
      undifference(y, pointForecasts, forecastVariances)

    SarimaForecast(
      mean     = undiffForecasts,
      variance = undiffVariances,
      lower95  = undiffForecasts.zip(undiffVariances).map { case (f, v) =>
        f - 1.96 * math.sqrt(math.max(v, 0.0))
      },
      upper95  = undiffForecasts.zip(undiffVariances).map { case (f, v) =>
        f + 1.96 * math.sqrt(math.max(v, 0.0))
      }
    )
  }

  private def difference(y: Array[Double]): Array[Double] = {
    var yd = y
    for (_ <- 0 until sarimaModel.d)
      yd = yd.zip(yd.tail).map { case (a, b) => b - a }
    for (_ <- 0 until sarimaModel.D)
      yd = yd.zipWithIndex
        .filter(_._2 >= sarimaModel.s)
        .map { case (v, i) => v - yd(i - sarimaModel.s) }
    yd
  }

  private def undifference(
                            y:         Array[Double],
                            forecasts: Array[Double],
                            variances: Array[Double]
                          ): (Array[Double], Array[Double]) = {
    var f = forecasts.clone()
    var v = variances.clone()

    if (sarimaModel.d > 0) {
      val levels = Array.fill(sarimaModel.d)(Array.empty[Double])
      levels(0) = y
      for (i <- 1 until sarimaModel.d)
        levels(i) = levels(i-1).zip(levels(i-1).tail).map { case (a, b) => b - a }

      for (i <- (0 until sarimaModel.d).reverse) {
        val lastObs = levels(i).last
        f = f.scanLeft(lastObs)(_ + _).tail
        v = v.scanLeft(0.0)(_ + _).tail
      }
    }

    if (sarimaModel.D > 0) {
      val s          = sarimaModel.s
      val lastSeason = y.takeRight(s)

      for (_ <- 0 until sarimaModel.D) {
        val extended = lastSeason ++ f
        f = (s until extended.length).map(i => extended(i) + extended(i - s)).toArray
        val extV = Array.fill(s)(0.0) ++ v
        v = (s until extV.length).map(i => extV(i) + extV(i - s)).toArray
      }
    }

    (f, v)
  }
}

case class SarimaForecast(
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