package series.filters

import org.scalatest.funsuite.AnyFunSuite

class ButterworthFilterTest extends AnyFunSuite {
  val rng = new scala.util.Random(42)
  def series(n: Int) = Array.fill(n)(rng.nextGaussian())

  test("lowpass output has same length as input") {
    val y = series(200)
    val r = ButterworthFilter.lowPass(2, 0.3).filter(y)
    assert(r.filtered.length == 200)
  }

  test("highpass output has same length") {
    val y = series(200)
    val r = ButterworthFilter.highPass(2, 0.3).filter(y)
    assert(r.filtered.length == 200)
  }

  test("bandpass output has same length") {
    val y = series(200)
    val r = ButterworthFilter.bandPass(2, 0.1, 0.4).filter(y)
    assert(r.filtered.length == 200)
  }

  test("effective order is double one-sided order") {
    val r = ButterworthFilter.lowPass(3, 0.3).filter(series(200))
    assert(r.effectiveOrder == 6)
  }

  test("lowpass attenuates high frequencies") {
    val n    = 500
    val low  = Array.tabulate(n)(t => math.sin(2 * math.Pi * 0.05 * t))
    val high = Array.tabulate(n)(t => math.sin(2 * math.Pi * 0.45 * t))
    val y    = low.zip(high).map { case (l, h) => l + h }
    val r    = ButterworthFilter.lowPass(4, 0.2).filter(y)
    // After low-pass: correlation with low-freq component should be high
    val corrWithLow = correlation(r.filtered, low)
    assert(corrWithLow > 0.5)
  }

  def correlation(x: Array[Double], y: Array[Double]): Double = {
    val n = x.length; val mx = x.sum/n; val my = y.sum/n
    val num = x.zip(y).map { case (xi, yi) => (xi-mx)*(yi-my) }.sum
    val sx  = math.sqrt(x.map(xi => (xi-mx)*(xi-mx)).sum)
    val sy  = math.sqrt(y.map(yi => (yi-my)*(yi-my)).sum)
    num / math.max(sx * sy, 1e-10)
  }

  test("filtered values are finite") {
    val y = series(200)
    val r = ButterworthFilter.lowPass(2, 0.3).filter(y)
    assert(r.filtered.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("toString is readable") {
    val r = ButterworthFilter.lowPass(2, 0.3).filter(series(200))
    assert(r.toString.contains("Butterworth"))
  }
}
