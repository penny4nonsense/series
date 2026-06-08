package series.vecm

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class VecmDiagnosticsTest extends AnyFunSuite {

  def simulateCointegrated(n: Int, seed: Long = 42): DenseMatrix[Double] = {
    val rng = new scala.util.Random(seed)
    val Y   = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      Y(t, 0) = Y(t-1, 0) + rng.nextGaussian() * 0.1
      Y(t, 1) = Y(t, 0) + rng.nextGaussian() * 0.1
    }
    Y
  }

  def buildDiag(n: Int = 500): VecmDiagnostics = {
    val Y   = simulateCointegrated(n)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    new VecmDiagnostics(fit)
  }

  test("Ljung-Box statistic is positive") {
    val diag = buildDiag()
    val lb   = diag.ljungBox(10)
    assert(lb.statistic > 0.0)
  }

  test("Ljung-Box p-value is in [0,1]") {
    val diag = buildDiag()
    val lb   = diag.ljungBox(10)
    assert(lb.pValue >= 0.0 && lb.pValue <= 1.0)
  }

  test("Ljung-Box does not reject for well-specified VECM") {
    val diag = buildDiag(1000)
    val lb   = diag.ljungBox(10)
    assert(lb.pValue > 0.001)
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

  test("ARCH-LM returns one result per equation") {
    val diag = buildDiag()
    val arch = diag.archLM(5)
    assert(arch.length == 2)
  }

  test("ARCH-LM statistics are positive") {
    val diag = buildDiag()
    val arch = diag.archLM(5)
    assert(arch.forall(_.statistic > 0.0))
  }

  test("ARCH-LM p-values are in [0,1]") {
    val diag = buildDiag()
    val arch = diag.archLM(5)
    assert(arch.forall(a => a.pValue >= 0.0 && a.pValue <= 1.0))
  }

  test("persistence convergence is true for valid VECM") {
    val diag = buildDiag(1000)
    assert(diag.persistenceConvergence(50))
  }

  test("report contains all test names") {
    val diag = buildDiag()
    val r    = diag.report()
    assert(r.contains("Ljung-Box"))
    assert(r.contains("Jarque-Bera"))
    assert(r.contains("ARCH-LM"))
    assert(r.contains("Persistence"))
  }
}