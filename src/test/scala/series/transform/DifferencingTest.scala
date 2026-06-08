package series.transform

import org.scalatest.funsuite.AnyFunSuite

class DifferencingTest extends AnyFunSuite {

  val rng = new scala.util.Random(42)
  def randSeries(n: Int) = Array.fill(n)(rng.nextGaussian())

  // ── Non-seasonal ───────────────────────────────────────────────────────────

  test("diff reduces length by d") {
    val y = randSeries(100)
    assert(Differencing.diff(y, 1).length == 99)
    assert(Differencing.diff(y, 2).length == 98)
    assert(Differencing.diff(y, 0).length == 100)
  }

  test("diff d=1 computes first differences") {
    val y  = Array(1.0, 3.0, 6.0, 10.0)
    val dy = Differencing.diff(y, 1)
    assert(dy.sameElements(Array(2.0, 3.0, 4.0)))
  }

  test("diff d=2 computes second differences") {
    val y   = Array(1.0, 3.0, 6.0, 10.0)
    val d2y = Differencing.diff(y, 2)
    assert(d2y.sameElements(Array(1.0, 1.0)))
  }

  test("undiff inverts diff d=1") {
    val y      = randSeries(100)
    val dy     = Differencing.diff(y, 1)
    val inits  = y.take(1)
    val yHat   = Differencing.undiff(dy, inits, 1)
    y.zip(yHat).foreach { case (a, b) => assert(math.abs(a - b) < 1e-10) }
  }

  test("undiff inverts diff d=2") {
    val y     = randSeries(100)
    val dy    = Differencing.diff(y, 2)
    val inits = Array(y(0), y(1) - y(0))
    val yHat  = Differencing.undiff(dy, inits, 2)
    y.zip(yHat).foreach { case (a, b) => assert(math.abs(a - b) < 1e-10) }
  }

  test("diff d=0 is identity") {
    val y = randSeries(50)
    assert(Differencing.diff(y, 0).sameElements(y))
  }

  // ── Seasonal ───────────────────────────────────────────────────────────────

  test("seasonalDiff reduces length by s*D") {
    val y = randSeries(100)
    assert(Differencing.seasonalDiff(y, 12, 1).length == 88)
    assert(Differencing.seasonalDiff(y, 4,  2).length == 92)
  }

  test("seasonalDiff D=1 subtracts lag-s values") {
    val y  = Array.tabulate(8)(i => i.toDouble)
    val dy = Differencing.seasonalDiff(y, 4, 1)
    assert(dy.sameElements(Array(4.0, 4.0, 4.0, 4.0)))
  }

  test("seasonalUndiff inverts seasonalDiff") {
    val y     = randSeries(100)
    val inits = y.take(12)
    val dy    = Differencing.seasonalDiff(y, 12, 1)
    val yHat  = Differencing.seasonalUndiff(dy, inits, 12, 1)
    y.zip(yHat).foreach { case (a, b) => assert(math.abs(a - b) < 1e-10) }
  }

  test("seasonalUndiff inverts D=2") {
    val y      = randSeries(100)
    // initials: first s values of original, then first s values after first seasonal diff
    val inits1 = y.take(4)  // first period of original
    val after1 = Differencing.seasonalDiff(y, 4, 1)
    val inits2 = after1.take(4)  // first period after one seasonal diff
    val inits  = inits1 ++ inits2  // combined: length s*D = 8
    val dy     = Differencing.seasonalDiff(y, 4, 2)
    val yHat   = Differencing.seasonalUndiff(dy, inits, 4, 2)
    y.zip(yHat).foreach { case (a, b) => assert(math.abs(a - b) < 1e-10) }
  }

  // ── Combined ───────────────────────────────────────────────────────────────

  test("diffCombined round-trips") {
    val y = randSeries(100)
    val (dy, sInits, nsInits) = Differencing.diffCombined(y, d=1, D=1, s=12)
    val yHat = Differencing.undiffCombined(dy, d=1, D=1, s=12, nsInits, sInits)
    y.zip(yHat).foreach { case (a, b) => assert(math.abs(a - b) < 1e-10) }
  }

  test("diffCombined returns correct length") {
    val y = randSeries(100)
    val (dy, _, _) = Differencing.diffCombined(y, d=1, D=1, s=12)
    assert(dy.length == 100 - 12 - 1)
  }

  // ── Fractional ─────────────────────────────────────────────────────────────

  test("fracDiff weight pi_0 is 1") {
    val r = Differencing.fracDiff(randSeries(50), d=0.4)
    assert(math.abs(r.weights(0) - 1.0) < 1e-10)
  }

  test("fracDiff weights decay toward zero") {
    val r = Differencing.fracDiff(randSeries(100), d=0.4, L=50)
    assert(math.abs(r.weights(50)) < math.abs(r.weights(1)))
  }

  test("fracDiff weights are negative for k>=1 when 0<d<1") {
    val r = Differencing.fracDiff(randSeries(100), d=0.4, L=10)
    assert(r.weights.tail.forall(_ < 0))
  }

  test("fracDiff MacKinnon-Davidson and Hosking give same weights") {
    val y  = randSeries(100)
    val r1 = Differencing.fracDiff(y, d=0.3, L=20, Differencing.MacKinnonDavidson)
    val r2 = Differencing.fracDiff(y, d=0.3, L=20, Differencing.Hosking)
    r1.weights.zip(r2.weights).foreach { case (a, b) =>
      assert(math.abs(a - b) < 1e-10)
    }
  }

  test("fracDiff series has correct length") {
    val y = randSeries(100)
    val r = Differencing.fracDiff(y, d=0.4, L=20)
    assert(r.series.length == 100 - 20)
  }

  test("fracDiff d=0 gives identity") {
    val y = randSeries(50)
    val r = Differencing.fracDiff(y, d=0.0, L=10)
    r.series.zip(y.drop(10)).foreach { case (a, b) =>
      assert(math.abs(a - b) < 1e-10)
    }
  }

  test("fracDiff d=1 approximates integer differencing") {
    val y    = randSeries(100)
    val fr   = Differencing.fracDiff(y, d=1.0, L=50)
    val intr = Differencing.diff(y, 1)
    // After dropping nDropped, should roughly match
    fr.series.zip(intr.drop(49)).foreach { case (a, b) =>
      assert(math.abs(a - b) < 0.05)
    }
  }

  test("fracDiff result toString is readable") {
    val r = Differencing.fracDiff(randSeries(50), d=0.4, L=10)
    val s = r.toString
    assert(s.contains("FracDiff"))
    assert(s.contains("MacKinnon-Davidson"))
  }

  test("Hosking toString shows method name") {
    val r = Differencing.fracDiff(randSeries(50), d=0.4, L=10, Differencing.Hosking)
    assert(r.toString.contains("Hosking"))
  }
}
