package series.stats

import org.scalatest.funsuite.AnyFunSuite

class AutoCorrelationTest extends AnyFunSuite {

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

  def simulateMA1(
                   n:      Int,
                   theta:  Double,
                   sigma2: Double,
                   seed:   Long = 42
                 ): Array[Double] = {
    val rng = new scala.util.Random(seed)
    val eps = Array.fill(n)(rng.nextGaussian() * math.sqrt(sigma2))
    val y   = Array.fill(n)(0.0)
    y(0) = eps(0)
    for (t <- 1 until n)
      y(t) = eps(t) + theta * eps(t-1)
    y
  }

  def simulateWhiteNoise(n: Int, seed: Long = 42): Array[Double] = {
    val rng = new scala.util.Random(seed)
    Array.fill(n)(rng.nextGaussian())
  }

  // --- ACF tests ---

  test("ACF returns correct number of lags") {
    val y = simulateAR1(200, phi=0.7, sigma2=1.0)
    assert(AutoCorrelation.acf(y, 10).length == 10)
  }

  test("ACF at lag 1 is approximately phi for AR(1)") {
    // For AR(1), ACF(k) = phi^k
    val y    = simulateAR1(1000, phi=0.7, sigma2=1.0)
    val acf  = AutoCorrelation.acf(y, 5)
    assert(math.abs(acf(0) - 0.7) < 0.05)
  }

  test("ACF decays geometrically for AR(1)") {
    val y   = simulateAR1(1000, phi=0.7, sigma2=1.0)
    val acf = AutoCorrelation.acf(y, 5)
    // Each ACF value should be approximately phi times the previous
    for (k <- 1 until 5)
      assert(math.abs(acf(k) / acf(k-1) - 0.7) < 0.1)
  }

  test("ACF cuts off after lag 1 for MA(1)") {
    val y   = simulateMA1(1000, theta=0.6, sigma2=1.0)
    val acf = AutoCorrelation.acf(y, 10)
    val bound = AutoCorrelation.confidenceBound(1000)
    // ACF at lag 1 should be significant
    assert(math.abs(acf(0)) > bound)
    // ACF at lags 2+ should be near zero
    assert(acf.tail.forall(a => math.abs(a) < 3 * bound))
  }

  test("ACF of white noise is near zero at all lags") {
    val y     = simulateWhiteNoise(1000)
    val acf   = AutoCorrelation.acf(y, 20)
    val bound = AutoCorrelation.confidenceBound(1000)
    // At most a few lags should exceed the bound by chance
    val significant = acf.count(a => math.abs(a) > bound)
    assert(significant <= 4)
  }

  test("ACF values are in [-1, 1]") {
    val y   = simulateAR1(200, phi=0.7, sigma2=1.0)
    val acf = AutoCorrelation.acf(y, 10)
    assert(acf.forall(a => a >= -1.0 && a <= 1.0))
  }

  test("ACF throws on maxLag >= series length") {
    val y = simulateAR1(10, phi=0.5, sigma2=1.0)
    assertThrows[IllegalArgumentException] {
      AutoCorrelation.acf(y, 10)
    }
  }

  // --- PACF tests ---

  test("PACF returns correct number of lags") {
    val y = simulateAR1(200, phi=0.7, sigma2=1.0)
    assert(AutoCorrelation.pacf(y, 10).length == 10)
  }

  test("PACF at lag 1 equals ACF at lag 1") {
    val y    = simulateAR1(200, phi=0.7, sigma2=1.0)
    val acf  = AutoCorrelation.acf(y, 1)
    val pacf = AutoCorrelation.pacf(y, 1)
    assert(math.abs(pacf(0) - acf(0)) < 1e-10)
  }

  test("PACF cuts off after lag 1 for AR(1)") {
    val y     = simulateAR1(1000, phi=0.7, sigma2=1.0)
    val pacf  = AutoCorrelation.pacf(y, 10)
    val bound = AutoCorrelation.confidenceBound(1000)
    // PACF at lag 1 should be significant
    assert(math.abs(pacf(0)) > bound)
    // PACF at lags 2+ should be near zero
    assert(pacf.tail.forall(p => math.abs(p) < 3 * bound))
  }

  test("PACF approximately equals phi at lag 1 for AR(1)") {
    val y    = simulateAR1(1000, phi=0.7, sigma2=1.0)
    val pacf = AutoCorrelation.pacf(y, 5)
    assert(math.abs(pacf(0) - 0.7) < 0.05)
  }

  test("PACF values are in [-1, 1]") {
    val y    = simulateAR1(200, phi=0.7, sigma2=1.0)
    val pacf = AutoCorrelation.pacf(y, 10)
    assert(pacf.forall(p => p >= -1.0 && p <= 1.0))
  }

  test("PACF of white noise is near zero at all lags") {
    val y     = simulateWhiteNoise(1000)
    val pacf  = AutoCorrelation.pacf(y, 20)
    val bound = AutoCorrelation.confidenceBound(1000)
    val significant = pacf.count(p => math.abs(p) > bound)
    assert(significant <= 4)
  }

  // --- Confidence bound tests ---

  test("confidence bound decreases with sample size") {
    val b100  = AutoCorrelation.confidenceBound(100)
    val b1000 = AutoCorrelation.confidenceBound(1000)
    assert(b100 > b1000)
  }

  test("confidence bound is positive") {
    assert(AutoCorrelation.confidenceBound(100) > 0.0)
  }

  // --- Significant lag tests ---

  test("significant ACF lags include lag 1 for MA(1)") {
    val y    = simulateMA1(1000, theta=0.6, sigma2=1.0)
    val lags = AutoCorrelation.significantACFLags(y, 10)
    assert(lags.contains(1))
  }

  test("significant PACF lags include lag 1 for AR(1)") {
    val y    = simulateAR1(1000, phi=0.7, sigma2=1.0)
    val lags = AutoCorrelation.significantPACFLags(y, 10)
    assert(lags.contains(1))
  }

  // --- Order suggestion tests ---

  test("suggestOrders suggests AR for AR(1) process") {
    val y = simulateAR1(1000, phi = 0.7, sigma2 = 1.0)
    val sug = AutoCorrelation.suggestOrders(y, 15)
    // AR process: PACF cuts off, so q should be 0 or p > 0
    assert(sug.p > 0 || sug.note.contains("AR") || sug.note.contains("ARMA"))
  }

  test("suggestOrders suggests MA for MA(1) process") {
    val y = simulateMA1(1000, theta = 0.6, sigma2 = 1.0)
    val sug = AutoCorrelation.suggestOrders(y, 15)
    // MA process: ACF cuts off, so p should be 0 or q > 0
    assert(sug.q > 0 || sug.note.contains("MA") || sug.note.contains("ARMA"))
  }

  test("suggestOrders returns white noise for iid series") {
    val y   = simulateWhiteNoise(1000)
    val sug = AutoCorrelation.suggestOrders(y, 15)
    assert(sug.note.contains("White noise"))
  }

  test("OrderSuggestion toString is readable") {
    val sug = OrderSuggestion(p=1, q=0, note="AR process suggested")
    assert(sug.toString.contains("ARIMA"))
    assert(sug.toString.contains("AR process suggested"))
  }
}