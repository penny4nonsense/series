package series.arima

import org.scalatest.funsuite.AnyFunSuite

class ArimaForecasterTest extends AnyFunSuite {

  def simulateAR1(n: Int, phi: Double, sigma2: Double, seed: Long = 42): Array[Double] = {
    val rng = new scala.util.Random(seed)
    val y   = Array.fill(n)(0.0)
    for (t <- 1 until n)
      y(t) = phi * y(t - 1) + rng.nextGaussian() * math.sqrt(sigma2)
    y
  }

  def fitAR1(y: Array[Double]): (ArimaModel, ArimaFit) = {
    val model     = ArimaModel(p = 1, d = 0, q = 0)
    val estimator = new ArimaEstimator(model)
    val fit       = estimator.fit(y)
    (model, fit)
  }

  test("forecast returns correct horizon length") {
    val y            = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val (model, fit) = fitAR1(y)
    val forecaster   = new ArimaForecaster(model, fit)
    val fc           = forecaster.forecast(y, h = 10)
    assert(fc.mean.length == 10)
    assert(fc.variance.length == 10)
    assert(fc.lower95.length == 10)
    assert(fc.upper95.length == 10)
  }

  test("forecast variances are positive and increasing") {
    val y            = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val (model, fit) = fitAR1(y)
    val forecaster   = new ArimaForecaster(model, fit)
    val fc           = forecaster.forecast(y, h = 10)
    // All variances positive
    assert(fc.variance.forall(_ > 0.0))
    // Variances should be non-decreasing for AR(1)
    fc.variance.zip(fc.variance.tail).foreach { case (v1, v2) =>
      assert(v2 >= v1 - 1e-10)
    }
  }

  test("prediction intervals are correctly ordered") {
    val y            = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val (model, fit) = fitAR1(y)
    val forecaster   = new ArimaForecaster(model, fit)
    val fc           = forecaster.forecast(y, h = 10)
    fc.mean.zip(fc.lower95).zip(fc.upper95).foreach { case ((m, l), u) =>
      assert(l < m && m < u)
    }
  }

  test("AR(1) forecasts decay toward zero for stationary process") {
    val y = simulateAR1(500, phi = 0.7, sigma2 = 1.0)
    val (model, fit) = fitAR1(y)
    val forecaster = new ArimaForecaster(model, fit)
    val fc = forecaster.forecast(y, h = 20)
    val shortHorizon = math.abs(fc.mean(0))
    val longHorizon = math.abs(fc.mean(19))
    assert(longHorizon <= shortHorizon + 0.01)
  }

  test("random walk forecasts are flat") {
    val rng = new scala.util.Random(42)
    val y = Array.fill(100)(rng.nextGaussian()).scanLeft(0.0)(_ + _).tail
    val model = ArimaModel(p = 0, d = 1, q = 0)
    val estimator = new ArimaEstimator(model)
    val fit = estimator.fit(y)
    val forecaster = new ArimaForecaster(model, fit)
    val fc = forecaster.forecast(y, h = 10)
    // Random walk: consecutive forecasts should be identical
    fc.mean.zip(fc.mean.tail).foreach { case (f1, f2) =>
      assert(math.abs(f1 - f2) < 1e-6)
    }
  }

  test("throws on non-positive horizon") {
    val y            = simulateAR1(100, phi = 0.5, sigma2 = 1.0)
    val (model, fit) = fitAR1(y)
    val forecaster   = new ArimaForecaster(model, fit)
    assertThrows[IllegalArgumentException] {
      forecaster.forecast(y, h = 0)
    }
  }

  test("toString produces readable table") {
    val y            = simulateAR1(100, phi = 0.5, sigma2 = 1.0)
    val (model, fit) = fitAR1(y)
    val forecaster   = new ArimaForecaster(model, fit)
    val fc           = forecaster.forecast(y, h = 5)
    val str          = fc.toString
    assert(str.contains("mean"))
    assert(str.contains("lower95"))
    assert(str.contains("upper95"))
  }
}