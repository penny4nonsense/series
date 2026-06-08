package series.svar

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite
import series.`var`.VarModel

class ShortRunIdentificationTest extends AnyFunSuite {

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

  test("Cholesky B0inv is lower triangular") {
    val svar   = buildSvar()
    val result = ShortRunIdentification.cholesky(svar, h=10)
    assert(math.abs(result.B0inv(0, 1)) < 1e-10)
  }

  test("Cholesky IRFs have correct horizon count") {
    val svar   = buildSvar()
    val result = ShortRunIdentification.cholesky(svar, h=10)
    assert(result.sirfs.length == 11)
  }

  test("Cholesky IRFs decay for stable VAR") {
    val svar   = buildSvar(1000)
    val result = ShortRunIdentification.cholesky(svar, h=50)
    val last   = result.sirfs(50)
    assert(norm(last.toDenseVector) < 0.1)
  }

  test("Cholesky sigma consistency") {
    val svar   = buildSvar()
    val result = ShortRunIdentification.cholesky(svar, h=10)
    assert(svar.verifySigma(result.B0inv))
  }

  test("Cholesky shocks have unit variance approximately") {
    val svar   = buildSvar(1000)
    val result = ShortRunIdentification.cholesky(svar, h=10)
    for (j <- 0 until 2) {
      val s = result.shocks(::, j).toArray
      val v = s.map(x => x*x).sum / s.length
      assert(math.abs(v - 1.0) < 0.3)
    }
  }

  test("FEVD proportions sum to 1") {
    val svar   = buildSvar()
    val result = ShortRunIdentification.cholesky(svar, h=10)
    val fevd   = result.fevd
    for (s <- 0 to 10; i <- 0 until 2) {
      val total = (0 until 2).map(j => fevd(s)(i)(j)).sum
      assert(math.abs(total - 1.0) < 1e-8)
    }
  }

  test("zero restrictions require k(k-1)/2 restrictions") {
    val svar = buildSvar()
    assertThrows[IllegalArgumentException] {
      ShortRunIdentification.zeroRestrictions(svar, Seq.empty, h=10)
    }
  }

  test("zero restrictions result is finite") {
    val svar   = buildSvar()
    val result = ShortRunIdentification.zeroRestrictions(
      svar, Seq((0, 1)), h=10
    )
    assert(result.sirfs.forall(m =>
      m.toArray.forall(v => !v.isNaN && !v.isInfinite)))
  }

  test("toString contains identification scheme") {
    val svar   = buildSvar()
    val result = ShortRunIdentification.cholesky(svar, h=10)
    assert(result.toString.contains("Cholesky"))
  }

  test("custom shock names appear in output") {
    val svar   = buildSvar()
    val result = ShortRunIdentification.cholesky(
      svar, h=10,
      shockNames = Some(Array("supply", "demand"))
    )
    assert(result.shockNames.contains("supply"))
    assert(result.shockNames.contains("demand"))
  }
}
