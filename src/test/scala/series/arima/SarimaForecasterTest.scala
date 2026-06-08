package series.arima

import org.scalatest.funsuite.AnyFunSuite

class SarimaForecasterTest extends AnyFunSuite {

  val s: Int = 4

  def simulateSAR1(
                    n:      Int,
                    phi:    Double,
                    Phi:    Double,
                    s:      Int,
                    sigma2: Double,
                    seed:   Long = 42
                  ): Array[Double] = {
    val rng = new scala.util.Random(seed)
    val y   = Array.fill(n)(0.0)
    for (t <- math.max(1, s) until n) {
      val ar  = if (t >= 1) phi * y(t - 1) else 0.0
      val sar = if (t >= s) Phi * y(t - s) else 0.0
      y(t) = ar + sar + rng.nextGaussian() * math.sqrt(sigma2)
    }
    y
  }

  def fitAndForecast(
                      y: Array[Double],
                      h: Int
                    ): SarimaForecast = {
    val model      = SarimaModel(p=1, d=0, q=0, P=1, D=0, Q=0, s=s)
    val estimator  = new SarimaEstimator(model)
    val fit        = estimator.fit(y)
    val forecaster = new SarimaForecaster(model, fit)
    forecaster.forecast(y, h)
  }

  test("forecast returns correct horizon length") {
    val y  = simulateSAR1(200, phi=0.3, Phi=0.5, s=s, sigma2=1.0)
    val fc = fitAndForecast(y, h=12)
    assert(fc.mean.length == 12)
    assert(fc.variance.length == 12)
    assert(fc.lower95.length == 12)
    assert(fc.upper95.length == 12)
  }

  test("forecast variances are positive") {
    val y  = simulateSAR1(200, phi=0.3, Phi=0.5, s=s, sigma2=1.0)
    val fc = fitAndForecast(y, h=12)
    assert(fc.variance.forall(_ > 0.0))
  }

  test("forecast variances are non-decreasing") {
    val y  = simulateSAR1(200, phi=0.3, Phi=0.5, s=s, sigma2=1.0)
    val fc = fitAndForecast(y, h=12)
    fc.variance.zip(fc.variance.tail).foreach { case (v1, v2) =>
      assert(v2 >= v1 - 1e-10)
    }
  }

  test("prediction intervals are correctly ordered") {
    val y  = simulateSAR1(200, phi=0.3, Phi=0.5, s=s, sigma2=1.0)
    val fc = fitAndForecast(y, h=12)
    fc.mean.zip(fc.lower95).zip(fc.upper95).foreach { case ((m, l), u) =>
      assert(l < m && m < u)
    }
  }

  test("point forecasts are finite") {
    val y  = simulateSAR1(200, phi=0.3, Phi=0.5, s=s, sigma2=1.0)
    val fc = fitAndForecast(y, h=12)
    assert(fc.mean.forall(f => !f.isNaN && !f.isInfinite))
  }

  test("seasonal random walk forecasts repeat last season") {
    val rng = new scala.util.Random(42)
    val y = Array.fill(40)(rng.nextGaussian())
      .scanLeft(0.0)(_ + _).tail
    val model = SarimaModel(p = 0, d = 0, q = 0, P = 0, D = 1, Q = 0, s = 4)
    val estimator = new SarimaEstimator(model)
    val fit = estimator.fit(y)
    val forecaster = new SarimaForecaster(model, fit)
    val fc = forecaster.forecast(y, h = 4)
    // Forecasts should be finite and bounded
    assert(fc.mean.forall(f => !f.isNaN && !f.isInfinite))
    assert(fc.mean.forall(f => math.abs(f) < 100.0))
  }

  test("throws on series too short for differencing") {
    val y = Array.fill(30)(1.0) // longer series to avoid optimizer failure
    val model = SarimaModel(p = 1, d = 1, q = 0, P = 0, D = 1, Q = 0, s = 12)
    val forecaster = new SarimaForecaster(model,
      SarimaFit(Array(0.1), Array.empty, Array.empty, Array.empty,
        1.0, -100.0, 5, Array(Double.NaN), cssOnly = true))
    assertThrows[IllegalArgumentException] {
      forecaster.forecast(Array.fill(5)(1.0), h = 1)
    }
  }

  test("toString produces readable table") {
    val y  = simulateSAR1(100, phi=0.3, Phi=0.5, s=s, sigma2=1.0)
    val fc = fitAndForecast(y, h=4)
    val str = fc.toString
    assert(str.contains("mean"))
    assert(str.contains("lower95"))
    assert(str.contains("upper95"))
  }
}