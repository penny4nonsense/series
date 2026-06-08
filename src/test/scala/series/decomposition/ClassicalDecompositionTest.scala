package series.decomposition

import org.scalatest.funsuite.AnyFunSuite

class ClassicalDecompositionTest extends AnyFunSuite {

  def seasonalSeries(n: Int, period: Int): Array[Double] =
    Array.tabulate(n) { t =>
      10.0 + 3.0 * math.sin(2 * math.Pi * t / period) + new scala.util.Random(42).nextGaussian() * 0.3
    }

  test("components have correct length") {
    val y = seasonalSeries(120, 12)
    val r = new ClassicalDecomposition(12).decompose(y)
    assert(r.trend.length == 120)
    assert(r.seasonal.length == 120)
    assert(r.irregular.length == 120)
  }

  test("additive: seasonal factors sum to zero") {
    val y = seasonalSeries(120, 12)
    val r = new ClassicalDecomposition(12, additive=true).decompose(y)
    val validS = r.seasonal.filter(!_.isNaN)
    val sum = validS.grouped(12).map(_.sum).toArray
    sum.foreach(s => assert(math.abs(s) < 0.01))
  }

  test("multiplicative: seasonal factors average to one") {
    val y = seasonalSeries(120, 12).map(_ + 20)  // ensure positive
    val r = new ClassicalDecomposition(12, additive=false).decompose(y)
    val period = r.seasonal.take(12)
    assert(math.abs(period.sum / 12 - 1.0) < 0.01)
  }

  test("seasonal strength is positive for seasonal series") {
    val y = seasonalSeries(120, 12)
    val r = new ClassicalDecomposition(12).decompose(y)
    assert(r.seasonalStrength > 0.0)
  }

  test("toString is readable") {
    val r = new ClassicalDecomposition(12).decompose(seasonalSeries(120, 12))
    assert(r.toString.contains("Classical Decomposition"))
  }
}
