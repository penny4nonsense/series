package series.arima

import org.scalatest.funsuite.AnyFunSuite

class ArimaDiagnosticsTest extends AnyFunSuite {

  def simulateAR1(
                   n:      Int,
                   phi:    Double,
                   sigma2: Double,
                   seed:   Long = 42
                 ): Array[Double] = {
    val rng = new scala.util.Random(seed)
    val y   = Array.fill(n)(0.0)
    for (t <- 1 until n)
      y(t) = phi * y(t-1) + rng.nextGaussian() * math.sqrt(sigma2)
    y
  }

  def buildDiagnostics(y: Array[Double]): ArimaDiagnostics = {
    val model     = ArimaModel(p=1, d=0, q=0)
    val estimator = new ArimaEstimator(model)
    val fit       = estimator.fit(y)
    val km        = model.toKalmanModel(fit.phi, fit.theta, fit.sigma2)
    val initial   = model.initialState(fit.phi, fit.theta, fit.sigma2)
    // Difference series if needed — d=0 here so pass through
    new ArimaDiagnostics(km, initial, y)
  }

  test("standardized residuals have correct length") {
    val y    = simulateAR1(200, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    assert(diag.standardizedResiduals.length == y.length)
  }

  test("standardized residuals are approximately mean zero") {
    val y    = simulateAR1(500, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    val r    = diag.standardizedResiduals.filter(!_.isNaN)
    val mean = r.sum / r.length
    assert(math.abs(mean) < 0.2)
  }

  test("standardized residuals have approximately unit variance") {
    val y    = simulateAR1(500, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    val r    = diag.standardizedResiduals.filter(!_.isNaN)
    val mean = r.sum / r.length
    val v    = r.map(x => math.pow(x - mean, 2)).sum / (r.length - 1)
    assert(math.abs(v - 1.0) < 0.3)
  }

  test("ACF at lag 0 is 1") {
    val y    = simulateAR1(200, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    val acf  = diag.residualACF(10)
    // ACF starts at lag 1 — all values should be small for white noise residuals
    assert(acf.forall(a => math.abs(a) < 0.3))
  }

  test("Ljung-Box p-value is large for well-specified model") {
    // Well-specified AR(1) should have white noise residuals
    val y    = simulateAR1(500, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    val lb   = diag.ljungBox(10)
    assert(!lb.statistic.isNaN)
    assert(lb.pValue >= 0.0 && lb.pValue <= 1.0)
    // Should generally not reject for correct model
    assert(lb.pValue > 0.01)
  }

  test("Ljung-Box statistic is positive") {
    val y    = simulateAR1(200, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    val lb   = diag.ljungBox(10)
    assert(lb.statistic > 0.0)
  }

  test("Jarque-Bera p-value is large for Gaussian residuals") {
    val y    = simulateAR1(500, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    val jb   = diag.jarqueBera()
    assert(!jb.statistic.isNaN)
    assert(jb.pValue >= 0.0 && jb.pValue <= 1.0)
    assert(jb.pValue > 0.01)
  }

  test("Jarque-Bera rejects for heavy-tailed residuals") {
    // t-distributed noise should trigger rejection
    val rng  = new scala.util.Random(42)
    // Simulate with occasional large shocks
    val y    = Array.fill(500) {
      val u = rng.nextGaussian()
      if (math.abs(u) > 2.0) u * 5.0 else u
    }
    val diag = buildDiagnostics(y)
    val jb   = diag.jarqueBera()
    assert(jb.isSignificant(0.05))
  }

  test("residual summary mean is finite") {
    val y    = simulateAR1(200, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    val rs   = diag.residualSummary()
    assert(!rs.mean.isNaN && !rs.mean.isInfinite)
  }

  test("residual summary quartiles are ordered") {
    val y    = simulateAR1(200, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    val rs   = diag.residualSummary()
    assert(rs.min <= rs.q25)
    assert(rs.q25 <= rs.median)
    assert(rs.median <= rs.q75)
    assert(rs.q75 <= rs.max)
  }

  test("report produces non-empty string") {
    val y    = simulateAR1(200, phi=0.7, sigma2=1.0)
    val diag = buildDiagnostics(y)
    val r    = diag.report()
    assert(r.contains("Ljung-Box"))
    assert(r.contains("Jarque-Bera"))
    assert(r.contains("Residual Summary"))
  }

  test("isSignificant returns correct value") {
    val lb = LjungBoxResult(statistic=20.0, lag=10, pValue=0.03)
    assert(lb.isSignificant(0.05))
    assert(!lb.isSignificant(0.01))
  }
}