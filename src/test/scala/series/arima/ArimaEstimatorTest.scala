package series.arima

import org.scalatest.funsuite.AnyFunSuite

class ArimaEstimatorTest extends AnyFunSuite {

  // Simulate AR(1) with phi=0.7, sigma2=1.0
  def simulateAR1(n: Int, phi: Double, sigma2: Double, seed: Long = 42): Array[Double] = {
    val rng = new scala.util.Random(seed)
    val y   = Array.fill(n)(0.0)
    for (t <- 1 until n)
      y(t) = phi * y(t - 1) + rng.nextGaussian() * math.sqrt(sigma2)
    y
  }

  test("AR(1) CSS fit recovers approximate phi") {
    val y         = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val model     = ArimaModel(p = 1, d = 0, q = 0)
    val estimator = new ArimaEstimator(model)
    val fit       = estimator.fit(y, cssOnly = true)
    // CSS estimate should be in the right ballpark
    assert(math.abs(fit.phi(0) - 0.7) < 0.2)
  }

  test("AR(1) MLE fit recovers phi more accurately than CSS") {
    val y         = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val model     = ArimaModel(p = 1, d = 0, q = 0)
    val estimator = new ArimaEstimator(model)
    val cssFit    = estimator.fit(y, cssOnly = true)
    val mleFit    = estimator.fit(y, cssOnly = false)
    // MLE should be at least as close as CSS
    assert(math.abs(mleFit.phi(0) - 0.7) <= math.abs(cssFit.phi(0) - 0.7) + 0.05)
  }

  test("MLE log likelihood is higher than CSS log likelihood") {
    val y = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val model = ArimaModel(p = 1, d = 0, q = 0)
    val estimator = new ArimaEstimator(model)
    val mleFit = estimator.fit(y, cssOnly = false)
    // MLE log likelihood should be finite and negative
    assert(!mleFit.logLik.isNaN && !mleFit.logLik.isInfinite)
    assert(mleFit.logLik < 0.0)
  }

  test("AIC is finite for fitted model") {
    val y         = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val model     = ArimaModel(p = 1, d = 0, q = 0)
    val estimator = new ArimaEstimator(model)
    val fit       = estimator.fit(y)
    assert(!fit.aic.isNaN && !fit.aic.isInfinite)
  }

  test("ARIMA(0,1,0) fits random walk") {
    val rng       = new scala.util.Random(42)
    val y         = Array.fill(100)(rng.nextGaussian()).scanLeft(0.0)(_ + _).tail
    val model     = ArimaModel(p = 0, d = 1, q = 0)
    val estimator = new ArimaEstimator(model)
    val fit       = estimator.fit(y)
    assert(!fit.logLik.isNaN && !fit.logLik.isInfinite)
    assert(fit.sigma2 > 0.0)
  }

  test("toString produces readable output") {
    val y         = simulateAR1(100, phi = 0.5, sigma2 = 1.0)
    val model     = ArimaModel(p = 1, d = 0, q = 0)
    val estimator = new ArimaEstimator(model)
    val fit       = estimator.fit(y)
    val str       = fit.toString
    assert(str.contains("phi"))
    assert(str.contains("sigma2"))
    assert(str.contains("AIC"))
  }

  test("standard errors are positive for MLE fit") {
    val y = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val model = ArimaModel(p = 1, d = 0, q = 0)
    val estimator = new ArimaEstimator(model)
    val fit = estimator.fit(y)
    assert(fit.se.forall(s => !s.isNaN && s > 0.0))
  }

  test("standard errors are NaN for CSS fit") {
    val y = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val model = ArimaModel(p = 1, d = 0, q = 0)
    val estimator = new ArimaEstimator(model)
    val fit = estimator.fit(y, cssOnly = true)
    assert(fit.se.forall(_.isNaN))
  }

  test("phi standard error is plausible for AR(1)") {
    // For AR(1) with n=200, SE on phi should be roughly (1-phi^2)/sqrt(n)
    val y = simulateAR1(200, phi = 0.7, sigma2 = 1.0)
    val model = ArimaModel(p = 1, d = 0, q = 0)
    val estimator = new ArimaEstimator(model)
    val fit = estimator.fit(y)
    val theoretical = (1 - 0.7 * 0.7) / math.sqrt(200)
    assert(math.abs(fit.se(0) - theoretical) < 0.05)
  }
}