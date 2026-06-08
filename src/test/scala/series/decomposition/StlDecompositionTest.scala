package series.decomposition

import org.scalatest.funsuite.AnyFunSuite

class StlDecompositionTest extends AnyFunSuite {

  def seasonalSeries(n: Int, period: Int, seed: Long = 42): Array[Double] = {
    val rng = new scala.util.Random(seed)
    Array.tabulate(n) { t =>
      10.0 + 0.05 * t +                                 // trend
      3.0 * math.sin(2 * math.Pi * t / period) +        // seasonal
      rng.nextGaussian() * 0.5                           // noise
    }
  }

  test("components have correct length") {
    val y = seasonalSeries(120, 12)
    val r = new StlDecomposition(12).decompose(y)
    assert(r.trend.length == 120)
    assert(r.seasonal.length == 120)
    assert(r.remainder.length == 120)
  }

  test("components sum to original") {
    val y = seasonalSeries(120, 12)
    val r = new StlDecomposition(12).decompose(y)
    y.zip((0 until 120).map(t => r.trend(t) + r.seasonal(t) + r.remainder(t))).foreach {
      case (yi, sum) => assert(math.abs(yi - sum) < 1e-6)
    }
  }

  test("seasonal strength is high for seasonal series") {
    val y = seasonalSeries(240, 12)
    val r = new StlDecomposition(12).decompose(y)
    assert(r.seasonalStrength > 0.5)
  }

  test("trend strength is positive") {
    val y = seasonalSeries(120, 12)
    val r = new StlDecomposition(12).decompose(y)
    assert(r.trendStrength >= 0.0)
  }

  test("seasonal sum per period is near zero") {
    val y = seasonalSeries(240, 12)
    val r = new StlDecomposition(12).decompose(y)
    assert(math.abs(r.seasonalSumCheck) < 0.5)
  }

  test("remainder values are finite") {
    val y = seasonalSeries(120, 12)
    val r = new StlDecomposition(12).decompose(y)
    assert(r.remainder.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("robust flag is recorded") {
    val y  = seasonalSeries(120, 12)
    val r1 = new StlDecomposition(12, robust = true).decompose(y)
    val r2 = new StlDecomposition(12, robust = false).decompose(y)
    assert(r1.robust)
    assert(!r2.robust)
    assert(r1.weights.isDefined)
    assert(r2.weights.isEmpty)
  }

  test("toString is readable") {
    val r = new StlDecomposition(12).decompose(seasonalSeries(120, 12))
    assert(r.toString.contains("STL"))
  }
}
