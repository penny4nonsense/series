package series.garch

import org.scalatest.funsuite.AnyFunSuite

class GarchForecasterTest extends AnyFunSuite {

  def simulateGarch11(
    n:      Int,
    omega:  Double,
    alpha:  Double,
    beta:   Double,
    seed:   Long = 42
  ): Array[Double] = {
    val rng    = new scala.util.Random(seed)
    val eps    = Array.fill(n)(0.0)
    val sigma2 = Array.fill(n)(omega / (1.0 - alpha - beta))
    for (t <- 1 until n) {
      sigma2(t) = omega + alpha * eps(t-1) * eps(t-1) + beta * sigma2(t-1)
      eps(t)    = rng.nextGaussian() * math.sqrt(sigma2(t))
    }
    eps
  }

  def buildFit(n: Int = 500): GarchFit = {
    val eps = simulateGarch11(n, 0.01, 0.1, 0.8)
    GarchModel(p = 1, q = 1).fit(eps)
  }

  test("forecast returns correct horizon length") {
    val f = buildFit().forecast(10)
    assert(f.varianceForecasts.length == 10)
    assert(f.volatilityForecasts.length == 10)
  }

  test("variance forecasts are positive") {
    val f = buildFit().forecast(10)
    assert(f.varianceForecasts.forall(_ > 0.0))
  }

  test("volatility forecasts are positive") {
    val f = buildFit().forecast(10)
    assert(f.volatilityForecasts.forall(_ > 0.0))
  }

  test("volatility is sqrt of variance") {
    val f = buildFit().forecast(10)
    f.varianceForecasts.zip(f.volatilityForecasts).foreach { case (v, vol) =>
      assert(math.abs(math.sqrt(v) - vol) < 1e-10)
    }
  }

  test("long horizon forecasts converge to unconditional variance") {
    val f = buildFit(1000).forecast(100)
    val lastVar = f.varianceForecasts.last
    assert(math.abs(lastVar - f.unconditionalVariance) < 0.01)
  }

  test("variance forecasts are finite") {
    val f = buildFit().forecast(20)
    assert(f.varianceForecasts.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("throws on non-positive horizon") {
    assertThrows[IllegalArgumentException] {
      buildFit().forecast(0)
    }
  }

  test("unconditional variance is positive") {
    val f = buildFit().forecast(5)
    assert(f.unconditionalVariance > 0.0)
  }

  test("forecaster can also be constructed directly from fit") {
    val fit = buildFit()
    val f   = new GarchForecaster(fit).forecast(5)
    assert(f.varianceForecasts.length == 5)
  }

  test("toString produces readable table") {
    val f   = buildFit().forecast(5)
    val str = f.toString
    assert(str.contains("variance"))
    assert(str.contains("volatility"))
    assert(str.contains("Unconditional"))
  }
}
