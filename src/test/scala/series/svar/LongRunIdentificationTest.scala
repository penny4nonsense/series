package series.svar

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite
import series.`var`.VarModel

class LongRunIdentificationTest extends AnyFunSuite {

  def buildSvar(n: Int = 500, seed: Long = 42): SvarModel = {
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

  test("Blanchard-Quah long-run multiplier is lower triangular") {
    val svar   = buildSvar()
    val result = LongRunIdentification.blanchardQuah(svar, h=20)
    val d      = svar.longRunMultiplier * result.B0inv
    assert(math.abs(d(0, 1)) < 1e-4)
  }

  test("Blanchard-Quah IRFs have correct horizon count") {
    val svar   = buildSvar()
    val result = LongRunIdentification.blanchardQuah(svar, h=20)
    assert(result.sirfs.length == 21)
  }

  test("Blanchard-Quah IRFs are finite") {
    val svar   = buildSvar()
    val result = LongRunIdentification.blanchardQuah(svar, h=20)
    result.sirfs.foreach { m =>
      assert(m.toArray.forall(v => !v.isNaN && !v.isInfinite))
    }
  }

  test("Blanchard-Quah sigma consistency") {
    val svar   = buildSvar()
    val result = LongRunIdentification.blanchardQuah(svar, h=20)
    assert(svar.verifySigma(result.B0inv, tol=1e-4))
  }

  test("FEVD sums to 1 for Blanchard-Quah") {
    val svar   = buildSvar()
    val result = LongRunIdentification.blanchardQuah(svar, h=10)
    val fevd   = result.fevd
    for (s <- 0 to 10; i <- 0 until 2) {
      val total = (0 until 2).map(j => fevd(s)(i)(j)).sum
      assert(math.abs(total - 1.0) < 1e-8)
    }
  }

  test("identification string is correct") {
    val svar   = buildSvar()
    val result = LongRunIdentification.blanchardQuah(svar, h=10)
    assert(result.identification.contains("Blanchard-Quah"))
  }

  test("zero restrictions require k(k-1)/2 restrictions") {
    val svar = buildSvar()
    assertThrows[IllegalArgumentException] {
      LongRunIdentification.zeroRestrictions(svar, Seq.empty, h=10)
    }
  }

  test("custom shock names appear in result") {
    val svar   = buildSvar()
    val result = LongRunIdentification.blanchardQuah(
      svar, h=10,
      shockNames = Some(Array("supply", "demand"))
    )
    assert(result.shockNames.contains("supply"))
  }
}
