package series.garch

import org.scalatest.funsuite.AnyFunSuite

class GarchModelTest extends AnyFunSuite {

  // Simulate GARCH(1,1) series
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

  test("requires at least one of p or q positive") {
    assertThrows[IllegalArgumentException] {
      GarchModel(0, 0)
    }
  }

  test("ARCH(1) is valid") {
    val model = GarchModel(p=0, q=1)
    assert(model.nParams == 2)
  }

  test("GARCH(1,1) has correct nParams") {
    val model = GarchModel(p=1, q=1)
    assert(model.nParams == 3)
  }

  test("conditional variances have correct length") {
    val model = GarchModel(p=1, q=1)
    val eps   = simulateGarch11(200, 0.01, 0.1, 0.8)
    val sv    = model.conditionalVariances(eps, 0.01, Array(0.1), Array(0.8))
    assert(sv.length == eps.length)
  }

  test("conditional variances are positive") {
    val model = GarchModel(p=1, q=1)
    val eps   = simulateGarch11(200, 0.01, 0.1, 0.8)
    val sv    = model.conditionalVariances(eps, 0.01, Array(0.1), Array(0.8))
    assert(sv.forall(_ > 0.0))
  }

  test("log likelihood is finite for valid parameters") {
    val model = GarchModel(p=1, q=1)
    val eps   = simulateGarch11(200, 0.01, 0.1, 0.8)
    val ll    = model.logLikelihood(eps, 0.01, Array(0.1), Array(0.8))
    assert(!ll.isNaN && !ll.isInfinite)
  }

  test("GARCH(1,1) fit recovers approximate omega") {
    val eps   = simulateGarch11(1000, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    assert(math.abs(fit.omega - 0.01) < 0.02)
  }

  test("GARCH(1,1) fit recovers approximate alpha") {
    val eps   = simulateGarch11(1000, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    assert(math.abs(fit.alpha(0) - 0.1) < 0.08)
  }

  test("GARCH(1,1) fit recovers approximate beta") {
    val eps   = simulateGarch11(1000, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    assert(math.abs(fit.beta(0) - 0.8) < 0.1)
  }

  test("persistence is less than 1 for stationary GARCH") {
    val eps   = simulateGarch11(1000, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    assert(fit.persistence < 1.0)
  }

  test("unconditional variance is positive") {
    val eps   = simulateGarch11(500, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    assert(fit.unconditionalVariance > 0.0)
  }

  test("standardized residuals have approximately unit variance") {
    val eps   = simulateGarch11(1000, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    val z     = fit.standardizedResiduals
    val v     = z.map(x => x * x).sum / z.length
    assert(math.abs(v - 1.0) < 0.2)
  }

  test("AIC BIC AICc are finite") {
    val eps   = simulateGarch11(500, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    assert(!fit.aic.isNaN  && !fit.aic.isInfinite)
    assert(!fit.bic.isNaN  && !fit.bic.isInfinite)
    assert(!fit.aicc.isNaN && !fit.aicc.isInfinite)
  }

  test("standard errors are positive") {
    val eps   = simulateGarch11(500, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    assert(!fit.omegaSE.isNaN && fit.omegaSE > 0.0)
    assert(fit.alphaSEs.forall(s => !s.isNaN && s > 0.0))
    assert(fit.betaSEs.forall(s => !s.isNaN && s > 0.0))
  }

  test("ARCH(1) fits correctly") {
    val rng = new scala.util.Random(42)
    val eps = Array.fill(500)(0.0)
    for (t <- 1 until 500)
      eps(t) = rng.nextGaussian() * math.sqrt(0.1 + 0.3 * eps(t-1) * eps(t-1))
    val model = GarchModel(p=0, q=1)
    val fit   = model.fit(eps)
    assert(!fit.logLik.isNaN)
    assert(fit.alpha(0) > 0.0)
  }

  test("toString produces readable output") {
    val eps   = simulateGarch11(200, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    val str   = fit.toString
    assert(str.contains("omega"))
    assert(str.contains("alpha1"))
    assert(str.contains("beta1"))
    assert(str.contains("persistence"))
  }
}