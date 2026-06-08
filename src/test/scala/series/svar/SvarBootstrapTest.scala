package series.svar

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite
import series.`var`.VarModel

class SvarBootstrapTest extends AnyFunSuite {

  def buildSvar(n: Int = 100, seed: Long = 42): SvarModel = {
    val A   = DenseMatrix((0.5, 0.1), (0.0, 0.4))
    val rng = new scala.util.Random(seed)
    val Y   = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      val yhat = A * Y(t-1, ::).t
      Y(t, 0) = yhat(0) + rng.nextGaussian() * 0.1
      Y(t, 1) = yhat(1) + rng.nextGaussian() * 0.1
    }
    val fit = VarModel(p=1, k=2).fit(Y)
    new SvarModel(fit)
  }

  test("bootstrap returns correct number of replications") {
    val svar   = buildSvar()
    val result = SvarBootstrap.bootstrapCholesky(svar, h=5, nBoot=20)
    assert(result.bootIRFs.length == 20)
  }

  test("confidence bands are ordered correctly") {
    val svar        = buildSvar()
    val result      = SvarBootstrap.bootstrapCholesky(svar, h=5, nBoot=20)
    val (low, high) = result.confidenceBand(0, 0)
    low.zip(high).foreach { case (l, h) => assert(l <= h) }
  }

  test("median IRF has correct length") {
    val svar   = buildSvar()
    val result = SvarBootstrap.bootstrapCholesky(svar, h=5, nBoot=20)
    val med    = result.medianIRF(0, 0)
    assert(med.length == 6)
  }

  test("bootstrap IRFs are finite") {
    val svar   = buildSvar()
    val result = SvarBootstrap.bootstrapCholesky(svar, h=5, nBoot=20)
    result.bootIRFs.foreach { sirfs =>
      sirfs.foreach { m =>
        assert(m.toArray.forall(v => !v.isNaN && !v.isInfinite))
      }
    }
  }

  test("wild bootstrap also runs correctly") {
    val svar   = buildSvar()
    val result = SvarBootstrap.bootstrapCholesky(
      svar, h=5, nBoot=20,
      scheme = SvarBootstrap.Wild
    )
    assert(result.bootIRFs.length == 20)
  }

  test("Blanchard-Quah bootstrap runs correctly") {
    val svar   = buildSvar()
    val result = SvarBootstrap.bootstrapBlanchardQuah(svar, h=5, nBoot=20)
    assert(result.bootIRFs.length == 20)
  }

  test("toString contains key fields") {
    val svar   = buildSvar()
    val result = SvarBootstrap.bootstrapCholesky(svar, h=5, nBoot=20)
    val str    = result.toString
    assert(str.contains("nBoot"))
    assert(str.contains("shock"))
  }
}
