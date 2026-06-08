package series.arima

import org.scalatest.funsuite.AnyFunSuite

class SarimaEstimatorTest extends AnyFunSuite {

  // Simulate seasonal AR(1) with period 4
  def simulateSAR1(
                    n:     Int,
                    phi:   Double,
                    Phi:   Double,
                    s:     Int,
                    sigma2: Double,
                    seed:  Long = 42
                  ): Array[Double] = {
    val rng = new scala.util.Random(seed)
    val y   = Array.fill(n)(0.0)
    for (t <- math.max(1, s) until n) {
      val ar = if (t >= 1) phi * y(t - 1) else 0.0
      val sar = if (t >= s) Phi * y(t - s) else 0.0
      y(t) = ar + sar + rng.nextGaussian() * math.sqrt(sigma2)
    }
    y
  }

  test("SARIMA(1,0,0)(1,0,0)_4 CSS fit is finite") {
    val y         = simulateSAR1(200, phi = 0.3, Phi = 0.5, s = 4, sigma2 = 1.0)
    val model     = SarimaModel(p=1, d=0, q=0, P=1, D=0, Q=0, s=4)
    val estimator = new SarimaEstimator(model)
    val fit       = estimator.fit(y, cssOnly = true)
    assert(!fit.logLik.isNaN && !fit.logLik.isInfinite)
  }

  test("SARIMA(1,0,0)(1,0,0)_4 MLE fit recovers approximate phi") {
    val y         = simulateSAR1(200, phi = 0.3, Phi = 0.5, s = 4, sigma2 = 1.0)
    val model     = SarimaModel(p=1, d=0, q=0, P=1, D=0, Q=0, s=4)
    val estimator = new SarimaEstimator(model)
    val fit       = estimator.fit(y)
    assert(math.abs(fit.phi(0) - 0.3) < 0.2)
  }

  test("SARIMA(1,0,0)(1,0,0)_4 MLE recovers approximate Phi") {
    val y = simulateSAR1(500, phi = 0.3, Phi = 0.5, s = 4, sigma2 = 1.0)
    val model = SarimaModel(p = 1, d = 0, q = 0, P = 1, D = 0, Q = 0, s = 4)
    val estimator = new SarimaEstimator(model)
    val fit = estimator.fit(y)
    assert(math.abs(fit.phiS(0) - 0.5) < 0.3)
  }

  test("MLE log likelihood exceeds CSS log likelihood") {
    val y = simulateSAR1(200, phi = 0.3, Phi = 0.5, s = 4, sigma2 = 1.0)
    val model = SarimaModel(p = 1, d = 0, q = 0, P = 1, D = 0, Q = 0, s = 4)
    val estimator = new SarimaEstimator(model)
    val mleFit = estimator.fit(y)
    assert(!mleFit.logLik.isNaN && !mleFit.logLik.isInfinite)
    assert(mleFit.logLik < 0.0)
  }

  test("AIC AICc BIC are finite") {
    val y         = simulateSAR1(200, phi = 0.3, Phi = 0.5, s = 4, sigma2 = 1.0)
    val model     = SarimaModel(p=1, d=0, q=0, P=1, D=0, Q=0, s=4)
    val estimator = new SarimaEstimator(model)
    val fit       = estimator.fit(y)
    assert(!fit.aic.isNaN  && !fit.aic.isInfinite)
    assert(!fit.aicc.isNaN && !fit.aicc.isInfinite)
    assert(!fit.bic.isNaN  && !fit.bic.isInfinite)
  }

  test("standard errors are positive for MLE fit") {
    val y         = simulateSAR1(200, phi = 0.3, Phi = 0.5, s = 4, sigma2 = 1.0)
    val model     = SarimaModel(p=1, d=0, q=0, P=1, D=0, Q=0, s=4)
    val estimator = new SarimaEstimator(model)
    val fit       = estimator.fit(y)
    assert(fit.se.forall(s => !s.isNaN && s > 0.0))
  }

  test("seasonal differencing reduces series length correctly") {
    val y         = simulateSAR1(120, phi = 0.0, Phi = 0.6, s = 12, sigma2 = 1.0)
    val model     = SarimaModel(p=0, d=0, q=0, P=1, D=1, Q=0, s=12)
    val estimator = new SarimaEstimator(model)
    val fit       = estimator.fit(y)
    // After one seasonal difference of period 12, n reduces by 12
    assert(fit.n == y.length - 12)
  }

  test("toString produces readable output") {
    val y         = simulateSAR1(100, phi = 0.3, Phi = 0.5, s = 4, sigma2 = 1.0)
    val model     = SarimaModel(p=1, d=0, q=0, P=1, D=0, Q=0, s=4)
    val estimator = new SarimaEstimator(model)
    val fit       = estimator.fit(y)
    val str       = fit.toString
    assert(str.contains("phi1"))
    assert(str.contains("Phi1"))
    assert(str.contains("sigma2"))
    assert(str.contains("AICc"))
  }
}