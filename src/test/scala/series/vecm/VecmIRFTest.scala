package series.vecm

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class VecmIRFTest extends AnyFunSuite {

  def simulateCointegrated(n: Int, seed: Long = 42): DenseMatrix[Double] = {
    val rng = new scala.util.Random(seed)
    val Y   = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      Y(t, 0) = Y(t-1, 0) + rng.nextGaussian() * 0.1
      Y(t, 1) = Y(t, 0) + rng.nextGaussian() * 0.1
    }
    Y
  }

  def buildIRF(n: Int = 500): VecmIRF = {
    val Y   = simulateCointegrated(n)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    new VecmIRF(fit)
  }

  test("orthogonal IRF has correct horizon count") {
    val irf = buildIRF()
    val r   = irf.orthogonalIRF(10)
    assert(r.length == 11)
  }

  test("orthogonal IRF values are finite") {
    val irf = buildIRF()
    val r   = irf.orthogonalIRF(10)
    r.foreach { m =>
      assert(m.toArray.forall(v => !v.isNaN && !v.isInfinite))
    }
  }

  test("generalized IRF has correct horizon count") {
    val irf = buildIRF()
    val r   = irf.generalizedIRF(10)
    assert(r.length == 11)
  }

  test("cumulative IRF is finite") {
    val irf = buildIRF()
    val cum = irf.cumulativeIRF(20)
    assert(cum.toArray.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("persistence profiles have correct dimensions") {
    val irf      = buildIRF()
    val profiles = irf.persistenceProfiles(20)
    assert(profiles.length == 1)
    assert(profiles(0).length == 21)
  }

  test("persistence profiles are positive") {
    val irf      = buildIRF()
    val profiles = irf.persistenceProfiles(20)
    profiles.foreach { p =>
      assert(p.forall(_ >= 0.0))
    }
  }

  test("persistence profiles converge for cointegrated system") {
    val irf      = buildIRF(1000)
    val profiles = irf.persistenceProfiles(100)
    profiles.foreach { p =>
      assert(p.last < p.head)
    }
  }

  test("FEVD proportions sum to 1") {
    val irf  = buildIRF()
    val fevd = irf.fevd(10)
    for (s <- 0 to 10; i <- 0 until 2) {
      val total = (0 until 2).map(j => fevd(s)(i)(j)).sum
      assert(math.abs(total - 1.0) < 1e-8)
    }
  }

  test("throws on non-positive horizon") {
    val irf = buildIRF()
    assertThrows[IllegalArgumentException] {
      irf.orthogonalIRF(0)
    }
  }
}