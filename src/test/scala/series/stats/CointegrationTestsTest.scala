package series.stats

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class CointegrationTestsTest extends AnyFunSuite {

  // Simulate cointegrated pair: y = x + stationary noise
  def cointegratedPair(
                        n:    Int,
                        seed: Long = 42
                      ): (Array[Double], Array[Double]) = {
    val rng = new scala.util.Random(seed)
    val x   = Array.fill(n)(rng.nextGaussian()).scanLeft(0.0)(_ + _).tail
    val e   = Array.fill(n)(rng.nextGaussian() * 0.5)
    val y   = x.zip(e).map { case (xi, ei) => xi + ei }
    (y, x)
  }

  // Simulate non-cointegrated pair: two independent random walks
  def nonCointegratedPair(
                           n:    Int,
                           seed: Long = 42
                         ): (Array[Double], Array[Double]) = {
    val rng = new scala.util.Random(seed)
    val x   = Array.fill(n)(rng.nextGaussian()).scanLeft(0.0)(_ + _).tail
    val y   = Array.fill(n)(rng.nextGaussian()).scanLeft(0.0)(_ + _).tail
    (y, x)
  }

  // --- Engle-Granger ---

  test("EG statistic is finite") {
    val (y, x) = cointegratedPair(200)
    val r      = CointegrationTests.engleGranger(y, x)
    assert(!r.statistic.isNaN && !r.statistic.isInfinite)
  }

  test("EG rejects null for cointegrated pair") {
    val (y, x) = cointegratedPair(500)
    val r      = CointegrationTests.engleGranger(y, x)
    assert(r.rejectAt(0.05))
  }

  test("EG fails to reject for non-cointegrated pair") {
    val (y, x) = nonCointegratedPair(500)
    val r      = CointegrationTests.engleGranger(y, x)
    assert(!r.rejectAt(0.05))
  }

  test("EG recovers approximate beta") {
    // y = x + noise so beta should be near 1
    val (y, x) = cointegratedPair(500)
    val r      = CointegrationTests.engleGranger(y, x)
    assert(math.abs(r.beta - 1.0) < 0.1)
  }

  test("EG residuals have correct length") {
    val (y, x) = cointegratedPair(200)
    val r      = CointegrationTests.engleGranger(y, x)
    assert(r.residuals.length == y.length)
  }

  test("EG throws on unequal length series") {
    assertThrows[IllegalArgumentException] {
      CointegrationTests.engleGranger(Array(1.0, 2.0), Array(1.0))
    }
  }

  test("EG toString contains key fields") {
    val (y, x) = cointegratedPair(200)
    val r      = CointegrationTests.engleGranger(y, x)
    assert(r.toString.contains("beta"))
    assert(r.toString.contains("CV 5%"))
  }

  // --- Johansen ---

  test("Johansen returns correct number of eigenvalues") {
    val (y, x) = cointegratedPair(200)
    val Y      = DenseMatrix.tabulate(200, 2)((i, _) => 0.0)
    for (i <- 0 until 200) { Y(i, 0) = y(i); Y(i, 1) = x(i) }
    val r = CointegrationTests.johansen(Y, lags=1)
    assert(r.eigenvalues.length == 2)
    assert(r.traceStatistics.length == 2)
    assert(r.maxStatistics.length == 2)
  }

  test("Johansen eigenvalues are in [0, 1]") {
    val (y, x) = cointegratedPair(200)
    val Y      = DenseMatrix.tabulate(200, 2)((i, _) => 0.0)
    for (i <- 0 until 200) { Y(i, 0) = y(i); Y(i, 1) = x(i) }
    val r = CointegrationTests.johansen(Y, lags=1)
    assert(r.eigenvalues.forall(l => l >= 0.0 && l <= 1.0))
  }

  test("Johansen trace statistics are positive") {
    val (y, x) = cointegratedPair(200)
    val Y = DenseMatrix.tabulate(200, 2)((i, _) => 0.0)
    for (i <- 0 until 200) {
      Y(i, 0) = y(i); Y(i, 1) = x(i)
    }
    val r = CointegrationTests.johansen(Y, lags = 1)
    // First trace statistic should always be positive
    assert(r.traceStatistics(0) > 0.0)
    // All should be finite
    assert(r.traceStatistics.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("Johansen Array convenience wrapper works") {
    val (y, x) = cointegratedPair(200)
    val r      = CointegrationTests.johansen(Array(y, x), lags=1, regime="drift")
    assert(r.k == 2)
  }

  test("Johansen throws on invalid regime") {
    val (y, x) = cointegratedPair(200)
    assertThrows[IllegalArgumentException] {
      CointegrationTests.johansen(Array(y, x), lags=1, regime="constant")
    }
  }

  test("Johansen toString is readable") {
    val (y, x) = cointegratedPair(200)
    val r      = CointegrationTests.johansen(Array(y, x), lags=1, regime="drift")
    val str    = r.toString
    assert(str.contains("trace"))
    assert(str.contains("lambda"))
  }

  test("Johansen larger eigenvalue first") {
    val (y, x) = cointegratedPair(200)
    val r      = CointegrationTests.johansen(Array(y, x), lags=1, regime="drift")
    assert(r.eigenvalues(0) >= r.eigenvalues(1))
  }
}