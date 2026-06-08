package series.garch

import org.scalatest.funsuite.AnyFunSuite

class GarchDiagnosticsTest extends AnyFunSuite {

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

  def buildDiag(n: Int = 500): GarchDiagnostics = {
    val eps   = simulateGarch11(n, 0.01, 0.1, 0.8)
    val model = GarchModel(p=1, q=1)
    val fit   = model.fit(eps)
    new GarchDiagnostics(fit)
  }

  test("Ljung-Box on residuals is finite") {
    val diag = buildDiag()
    val lb   = diag.ljungBoxResiduals(10)
    assert(!lb.statistic.isNaN && !lb.statistic.isInfinite)
  }

  test("Ljung-Box on squared residuals is finite") {
    val diag = buildDiag()
    val lb   = diag.ljungBoxSquared(10)
    assert(!lb.statistic.isNaN && !lb.statistic.isInfinite)
  }

  test("ARCH-LM statistic is positive") {
    val diag = buildDiag()
    val lm   = diag.archLM(5)
    assert(lm.statistic > 0.0)
  }

  test("ARCH-LM p-value is in [0,1]") {
    val diag = buildDiag()
    val lm   = diag.archLM(5)
    assert(lm.pValue >= 0.0 && lm.pValue <= 1.0)
  }

  test("ARCH-LM does not reject for well-specified GARCH") {
    val diag = buildDiag(1000)
    val lm   = diag.archLM(5)
    assert(!lm.isSignificant(0.01))
  }

  test("Ljung-Box squared does not reject for well-specified GARCH") {
    val diag = buildDiag(1000)
    val lb   = diag.ljungBoxSquared(10)
    assert(!lb.isSignificant(0.01))
  }

  test("Jarque-Bera statistic is positive") {
    val diag = buildDiag()
    val jb   = diag.jarqueBera()
    assert(jb.statistic > 0.0)
  }

  test("Jarque-Bera p-value is in [0,1]") {
    val diag = buildDiag()
    val jb   = diag.jarqueBera()
    assert(jb.pValue >= 0.0 && jb.pValue <= 1.0)
  }

  test("residual summary quartiles are ordered") {
    val diag = buildDiag()
    val rs   = diag.residualSummary()
    assert(rs.min <= rs.q25)
    assert(rs.q25 <= rs.median)
    assert(rs.median <= rs.q75)
    assert(rs.q75 <= rs.max)
  }

  test("report contains all test names") {
    val diag = buildDiag()
    val r    = diag.report()
    assert(r.contains("Ljung-Box"))
    assert(r.contains("ARCH-LM"))
    assert(r.contains("Jarque-Bera"))
  }
}