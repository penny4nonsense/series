package series.vecm

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class VecmForecasterTest extends AnyFunSuite {

  def simulateCointegrated(n: Int, seed: Long = 42): DenseMatrix[Double] = {
    val rng = new scala.util.Random(seed)
    val Y   = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      Y(t, 0) = Y(t-1, 0) + rng.nextGaussian() * 0.1
      Y(t, 1) = Y(t, 0) + rng.nextGaussian() * 0.1
    }
    Y
  }

  def buildFit(n: Int = 300): VecmFit = {
    val Y = simulateCointegrated(n)
    VecmModel(k=2, p=2, r=1).fit(Y)
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

  test("MSE is non-decreasing") {
    val f = buildFit().forecast(10)
    for (j <- 0 until 2)
      f.mse.zip(f.mse.tail).foreach { case (m1, m2) =>
        assert(m2(j,j) >= m1(j,j) - 1e-10)
      }
  }

  test("cointegrating relationships are bounded for large horizon") {
    val f = buildFit(500).forecast(50)
    for (j <- 0 until 1) {
      val maxAbs = (0 until 50).map(t =>
        math.abs(f.cointRelationships(t, j))).max
      assert(maxAbs < 10.0)
    }
  }

  test("interval half widths are positive") {
    val f = buildFit().forecast(10)
    assert(f.intervalHalfWidths.toArray.forall(_ > 0.0))
  }

  test("throws on non-positive horizon") {
    assertThrows[IllegalArgumentException] {
      buildFit().forecast(0)
    }
  }

  test("forecaster can also be constructed directly from fit") {
    val fit = buildFit()
    val f   = new VecmForecaster(fit).forecast(5)
    assert(f.forecasts.rows == 5)
  }

  test("toString contains cointegrating relationships") {
    val f   = buildFit().forecast(5)
    val str = f.toString
    assert(str.contains("Cointegrating"))
    assert(str.contains("z1"))
  }
}
