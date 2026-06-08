package series.svar

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite
import series.`var`.VarModel
import series.svar.SignRestrictions.SignRestriction

class SignRestrictionsTest extends AnyFunSuite {

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

  val restrictions: Seq[SignRestriction] = Seq(
    SignRestriction(series=0, shock=0, horizon=0, sign=1),
    SignRestriction(series=1, shock=1, horizon=0, sign=1)
  )

  test("sign must be +1 or -1") {
    assertThrows[IllegalArgumentException] {
      SignRestriction(series=0, shock=0, horizon=0, sign=0)
    }
  }

  test("identify returns positive acceptance rate") {
    val svar   = buildSvar()
    val result = SignRestrictions.identify(svar, restrictions, h=10)
    assert(result.acceptanceRate > 0.0)
  }

  test("all admissible IRFs satisfy restrictions") {
    val svar   = buildSvar()
    val result = SignRestrictions.identify(svar, restrictions, h=10)
    result.admissibleIRFs.foreach { sirfs =>
      restrictions.foreach { r =>
        val irf = sirfs(r.horizon)(r.series, r.shock)
        if (r.sign == 1) assert(irf >= 0.0)
        else assert(irf <= 0.0)
      }
    }
  }

  test("median IRF has correct length") {
    val svar   = buildSvar()
    val result = SignRestrictions.identify(svar, restrictions, h=10)
    val med    = result.medianIRF(0, 0)
    assert(med.length == 11)
  }

  test("IRF bands are ordered correctly") {
    val svar        = buildSvar()
    val result      = SignRestrictions.identify(svar, restrictions, h=10)
    val (low, high) = result.irfBands(0, 0)
    low.zip(high).foreach { case (l, h) => assert(l <= h) }
  }

  test("FEVD proportions sum to 1") {
    val svar   = buildSvar()
    val result = SignRestrictions.identify(svar, restrictions, h=10)
    val fevd   = result.fevd
    for (s <- 0 to 10; i <- 0 until 2) {
      val total = (0 until 2).map(j => fevd(s)(i)(j)).sum
      assert(math.abs(total - 1.0) < 1e-6)
    }
  }

  test("acceptance rate is in (0, 1]") {
    val svar   = buildSvar()
    val result = SignRestrictions.identify(svar, restrictions, h=10)
    assert(result.acceptanceRate > 0.0 && result.acceptanceRate <= 1.0)
  }

  test("IRF distribution has correct dimensions") {
    val svar   = buildSvar()
    val result = SignRestrictions.identify(svar, restrictions, h=10)
    val dist   = result.irfDistribution(0, 0)
    assert(dist.length == result.nAccepted)
    assert(dist(0).length == 11)
  }

  test("throws on empty restrictions") {
    val svar = buildSvar()
    assertThrows[IllegalArgumentException] {
      SignRestrictions.identify(svar, Seq.empty, h=10)
    }
  }

  test("toString contains acceptance rate") {
    val svar   = buildSvar()
    val result = SignRestrictions.identify(svar, restrictions, h=10)
    assert(result.toString.contains("Acceptance rate"))
  }

  test("custom shock names appear in output") {
    val svar   = buildSvar()
    val result = SignRestrictions.identify(
      svar, restrictions, h=10,
      shockNames = Some(Array("supply", "demand"))
    )
    assert(result.toString.contains("supply"))
  }
}
