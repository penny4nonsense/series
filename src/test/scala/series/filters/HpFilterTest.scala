package series.filters

import org.scalatest.funsuite.AnyFunSuite

class HpFilterTest extends AnyFunSuite {
  val rng = new scala.util.Random(42)

  def randWalk(n: Int): Array[Double] = {
    val y = new Array[Double](n)
    for (t <- 1 until n) y(t) = y(t-1) + rng.nextGaussian()
    y
  }

  test("trend has correct length") {
    val y = randWalk(100)
    val r = new HpFilter(1600).filter(y)
    assert(r.trend.length == 100)
  }

  test("cycle has correct length") {
    val y = randWalk(100)
    val r = new HpFilter(1600).filter(y)
    assert(r.cycle.length == 100)
  }

  test("trend + cycle = original") {
    val y = randWalk(100)
    val r = new HpFilter(1600).filter(y)
    y.zip(r.trend.zip(r.cycle)).foreach { case (yi, (ti, ci)) =>
      assert(math.abs(yi - ti - ci) < 1e-8)
    }
  }

  test("lambda=0 gives trend = original") {
    val y = randWalk(20)
    val r = new HpFilter(0.0001).filter(y)
    y.zip(r.trend).foreach { case (yi, ti) =>
      assert(math.abs(yi - ti) < 0.01)
    }
  }

  test("large lambda gives near-linear trend") {
    val y = randWalk(100)
    val r = new HpFilter(1e10).filter(y)
    // Trend should be nearly linear: second differences near zero
    val d2 = (2 until 100).map(t => r.trend(t) - 2*r.trend(t-1) + r.trend(t-2))
    assert(d2.forall(v => math.abs(v) < 0.1))
  }

  test("cycle mean is near zero") {
    val y = randWalk(200)
    val r = new HpFilter(1600).filter(y)
    assert(math.abs(r.cycle.sum / 200) < 0.5)
  }

  test("variance ratio is positive") {
    val y = randWalk(100)
    val r = new HpFilter(1600).filter(y)
    assert(r.varianceRatio > 0)
  }

  test("toString is readable") {
    val r = new HpFilter(1600).filter(randWalk(100))
    assert(r.toString.contains("HP Filter"))
  }
}
