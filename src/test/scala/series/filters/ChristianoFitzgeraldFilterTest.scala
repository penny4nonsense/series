package series.filters

import org.scalatest.funsuite.AnyFunSuite

class ChristianoFitzgeraldFilterTest extends AnyFunSuite {
  val rng = new scala.util.Random(42)
  def series(n: Int) = Array.fill(n)(rng.nextGaussian())

  test("symmetric output has same length as input") {
    val y = series(100)
    val r = new ChristianoFitzgeraldFilter(6, 32).filterSymmetric(y)
    assert(r.filtered.length == 100)
  }

  test("asymmetric output has same length as input") {
    val y = series(100)
    val r = new ChristianoFitzgeraldFilter(6, 32).filterAsymmetric(y)
    assert(r.filtered.length == 100)
  }

  test("filtered values are finite") {
    val y = series(100)
    val r = new ChristianoFitzgeraldFilter(6, 32).filterSymmetric(y)
    assert(r.filtered.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("symmetric and asymmetric are correlated") {
    val y  = series(200)
    val rs = new ChristianoFitzgeraldFilter(6, 32).filterSymmetric(y).filtered
    val ra = new ChristianoFitzgeraldFilter(6, 32).filterAsymmetric(y).filtered
    val corr = correlation(rs, ra)
    assert(corr > 0.5)
  }

  test("toString is readable") {
    val r = new ChristianoFitzgeraldFilter(6, 32).filterSymmetric(series(100))
    assert(r.toString.contains("Christiano-Fitzgerald"))
  }

  def correlation(x: Array[Double], y: Array[Double]): Double = {
    val n  = x.length; val mx = x.sum/n; val my = y.sum/n
    val cov = x.zip(y).map { case (xi, yi) => (xi-mx)*(yi-my) }.sum / n
    val sx  = math.sqrt(x.map(xi => (xi-mx)*(xi-mx)).sum / n)
    val sy  = math.sqrt(y.map(yi => (yi-my)*(yi-my)).sum / n)
    cov / math.max(sx * sy, 1e-10)
  }
}
