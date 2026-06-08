package series.`var`

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class VarLagSelectionTest extends AnyFunSuite {

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

  test("returns criteria for all lags") {
    val Y   = simulateVAR1(200)
    val res = VarLagSelection.select(Y, maxLag=4)
    assert(res.criteria.size == 4)
  }

  test("selected lags are in valid range") {
    val Y   = simulateVAR1(200)
    val res = VarLagSelection.select(Y, maxLag=4)
    assert(res.aicLag  >= 1 && res.aicLag  <= 4)
    assert(res.bicLag  >= 1 && res.bicLag  <= 4)
    assert(res.hqicLag >= 1 && res.hqicLag <= 4)
  }

  test("BIC selects true lag for large sample") {
    val Y   = simulateVAR1(500)
    val res = VarLagSelection.select(Y, maxLag=4)
    assert(res.bicLag == 1)
  }

  test("criteria values are finite") {
    val Y   = simulateVAR1(200)
    val res = VarLagSelection.select(Y, maxLag=3)
    res.criteria.values.foreach { case (aic, bic, hqic) =>
      assert(!aic.isNaN && !bic.isNaN && !hqic.isNaN)
    }
  }

  test("toString contains selected lags") {
    val Y   = simulateVAR1(200)
    val res = VarLagSelection.select(Y, maxLag=3)
    val str = res.toString
    assert(str.contains("AIC"))
    assert(str.contains("BIC"))
    assert(str.contains("HQIC"))
    assert(str.contains("*"))
  }
}
