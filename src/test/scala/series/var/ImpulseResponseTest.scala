package series.`var`

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class ImpulseResponseTest extends AnyFunSuite {

  def simulateVAR1(n: Int, seed: Long = 42): DenseMatrix[Double] = {
    val A   = DenseMatrix((0.5, 0.1), (0.0, 0.4))
    val rng = new scala.util.Random(seed)
    val Y   = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      val yhat = A * Y(t-1, ::).t
      Y(t, 0) = yhat(0) + rng.nextGaussian() * 0.1
      Y(t, 1) = yhat(1) + rng.nextGaussian() * 0.1
    }
    Y
  }

  def buildIRF(n: Int = 500): ImpulseResponse = {
    val Y   = simulateVAR1(n)
    val fit = VarModel(p=1, k=2).fit(Y)
    new ImpulseResponse(fit)
  }

  test("orthogonal IRF returns correct horizon count") {
    val irf = buildIRF()
    val r   = irf.orthogonalIRF(10)
    assert(r.responses.length == 11)
  }

  test("IRF at horizon 0 is Cholesky factor") {
    val irf = buildIRF()
    val r   = irf.orthogonalIRF(10)
    assert(r.responses(0)(0, 1) == 0.0 || math.abs(r.responses(0)(0, 1)) < 1e-10)
  }

  test("orthogonal IRF decays to zero for stable VAR") {
    val irf = buildIRF(1000)
    val r   = irf.orthogonalIRF(50)
    val lastResp = r.responses(50)
    assert(norm(lastResp.toDenseVector) < 0.1)
  }

  test("generalized IRF returns correct horizon count") {
    val irf = buildIRF()
    val r   = irf.generalizedIRF(10)
    assert(r.responses.length == 11)
  }

  test("generalized IRF is finite") {
    val irf = buildIRF()
    val r   = irf.generalizedIRF(10)
    r.responses.foreach { m =>
      assert(m.toArray.forall(v => !v.isNaN && !v.isInfinite))
    }
  }

  test("response helper returns correct length") {
    val irf = buildIRF()
    val r   = irf.orthogonalIRF(10)
    assert(r.response(0, 0).length == 11)
  }

  test("FEVD proportions sum to 1 at each horizon") {
    val irf  = buildIRF()
    val fevd = irf.fevd(10)
    for (s <- 0 to 10; i <- 0 until 2) {
      val total = (0 until 2).map(j => fevd.proportions(s)(i)(j)).sum
      assert(math.abs(total - 1.0) < 1e-8)
    }
  }

  test("FEVD proportions are non-negative") {
    val irf  = buildIRF()
    val fevd = irf.fevd(10)
    for (s <- 0 to 10; i <- 0 until 2; j <- 0 until 2)
      assert(fevd.proportions(s)(i)(j) >= 0.0)
  }

  test("FEVD own-shock proportion is large at horizon 0") {
    val irf  = buildIRF(1000)
    val fevd = irf.fevd(10)
    assert(fevd.proportions(0)(0)(0) > 0.5)
  }

  test("IRF toString is readable") {
    val irf = buildIRF()
    val r   = irf.orthogonalIRF(5)
    val str = r.toString
    assert(str.contains("Shock"))
    assert(str.contains("Shock to series 1"))
  }

  test("FEVD toString is readable") {
    val irf  = buildIRF()
    val fevd = irf.fevd(5)
    val str  = fevd.toString
    assert(str.contains("Series"))
    assert(str.contains("shock"))
  }
}
