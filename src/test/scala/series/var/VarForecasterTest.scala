package series.`var`

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class VarForecasterTest extends AnyFunSuite {

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

  def buildFit(n: Int = 500): VarFit = {
    val Y = simulateVAR1(n)
    VarModel(p=1, k=2).fit(Y)
  }

  test("forecast returns correct dimensions") {
    val f = buildFit().forecast(10)
    assert(f.forecasts.rows == 10)
    assert(f.forecasts.cols == 2)
  }

  test("MSE matrices have correct dimensions") {
    val f = buildFit().forecast(10)
    assert(f.mse.length == 10)
    assert(f.mse(0).rows == 2 && f.mse(0).cols == 2)
  }

  test("forecast values are finite") {
    val f = buildFit().forecast(10)
    assert(f.forecasts.toArray.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("MSE matrices are positive definite") {
    val f = buildFit(1000).forecast(5)
    f.mse.foreach { m =>
      val eigs = eigSym(m)
      assert(eigs.eigenvalues.toArray.forall(_ > 0.0))
    }
  }

  test("MSE increases with horizon") {
    val f = buildFit().forecast(10)
    for (j <- 0 until 2)
      f.mse.zip(f.mse.tail).foreach { case (m1, m2) =>
        assert(m2(j,j) >= m1(j,j) - 1e-10)
      }
  }

  test("interval half widths are positive") {
    val f = buildFit().forecast(10)
    assert(f.intervalHalfWidths.toArray.forall(_ > 0.0))
  }

  test("stationary VAR forecasts converge toward mean") {
    val f = buildFit(1000).forecast(50)
    val lastFc = f.forecasts(49, ::)
    assert(math.abs(lastFc(0)) < 0.5)
    assert(math.abs(lastFc(1)) < 0.5)
  }

  test("throws on non-positive horizon") {
    assertThrows[IllegalArgumentException] {
      buildFit().forecast(0)
    }
  }

  test("forecaster can also be constructed directly from fit") {
    val fit = buildFit()
    val f   = new VarForecaster(fit).forecast(5)
    assert(f.forecasts.rows == 5)
  }

  test("toString produces readable output") {
    val f   = buildFit().forecast(5)
    val str = f.toString
    assert(str.contains("y1"))
    assert(str.contains("y2"))
  }
}
