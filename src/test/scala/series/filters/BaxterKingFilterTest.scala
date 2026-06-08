package series.filters

import org.scalatest.funsuite.AnyFunSuite

class BaxterKingFilterTest extends AnyFunSuite {
  val rng = new scala.util.Random(42)
  def series(n: Int) = Array.fill(n)(rng.nextGaussian())

  test("symmetric output has correct length") {
    val y = series(100)
    val r = new BaxterKingFilter(6, 32, 12).filterSymmetric(y)
    assert(r.filtered.length == 100 - 2 * 12)
  }

  test("asymmetric output has same length as input") {
    val y = series(100)
    val r = new BaxterKingFilter(6, 32, 12).filterAsymmetric(y)
    assert(r.filtered.length == 100)
  }

  test("weights sum to approximately zero") {
    val bk = new BaxterKingFilter(6, 32, 12)
    assert(math.abs(bk.weights.sum) < 1e-10)
  }

  test("weights are symmetric") {
    val bk = new BaxterKingFilter(6, 32, 12)
    val w  = bk.weights
    w.zip(w.reverse).foreach { case (a, b) =>
      assert(math.abs(a - b) < 1e-10)
    }
  }

  test("filtered values are finite") {
    val y = series(100)
    val r = new BaxterKingFilter(6, 32, 12).filterSymmetric(y)
    assert(r.filtered.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("nDropped equals K") {
    val r = new BaxterKingFilter(6, 32, 12).filterSymmetric(series(100))
    assert(r.nDropped == 12)
  }

  test("toString is readable") {
    val r = new BaxterKingFilter(6, 32, 12).filterSymmetric(series(100))
    assert(r.toString.contains("Baxter-King"))
  }
}
