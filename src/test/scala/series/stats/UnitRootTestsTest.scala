package series.stats

import org.scalatest.funsuite.AnyFunSuite

class UnitRootTestsTest extends AnyFunSuite {

  val seed: Long = 42

  def randomWalk(n: Int, seed: Long = 42): Array[Double] = {
    val rng = new scala.util.Random(seed)
    Array.fill(n)(rng.nextGaussian()).scanLeft(0.0)(_ + _).tail
  }

  def stationary(n: Int, phi: Double = 0.5, seed: Long = 42): Array[Double] = {
    val rng = new scala.util.Random(seed)
    val y   = Array.fill(n)(0.0)
    for (t <- 1 until n)
      y(t) = phi * y(t-1) + rng.nextGaussian()
    y
  }

  // --- ADF ---

  test("ADF statistic is finite") {
    val y = stationary(200)
    val r = UnitRootTests.adf(y)
    assert(!r.statistic.isNaN && !r.statistic.isInfinite)
  }

  test("ADF rejects unit root for stationary AR(1)") {
    val y = stationary(500, phi=0.3)
    val r = UnitRootTests.adf(y)
    assert(r.rejectAt(0.05))
  }

  test("ADF fails to reject for random walk") {
    val y = randomWalk(500)
    val r = UnitRootTests.adf(y)
    assert(!r.rejectAt(0.05))
  }

  test("ADF statistic is negative") {
    val y = stationary(200)
    val r = UnitRootTests.adf(y)
    assert(r.statistic < 0.0)
  }

  test("ADF works with all regimes") {
    val y = stationary(200)
    Seq("none", "drift", "trend").foreach { regime =>
      val r = UnitRootTests.adf(y, regime=regime)
      assert(!r.statistic.isNaN)
    }
  }

  test("ADF throws on invalid regime") {
    val y = stationary(200)
    assertThrows[IllegalArgumentException] {
      UnitRootTests.adf(y, regime="constant")
    }
  }

  test("ADF toString contains key fields") {
    val y = stationary(200)
    val r = UnitRootTests.adf(y)
    assert(r.toString.contains("t-statistic"))
    assert(r.toString.contains("CV 5%"))
  }

  // --- PP ---

  test("PP statistic is finite") {
    val y = stationary(200)
    val r = UnitRootTests.pp(y)
    assert(!r.statistic.isNaN && !r.statistic.isInfinite)
  }

  test("PP rejects unit root for stationary series") {
    val y = stationary(500, phi=0.3)
    val r = UnitRootTests.pp(y)
    assert(r.rejectAt(0.05))
  }

  test("PP fails to reject for random walk") {
    val y = randomWalk(500)
    val r = UnitRootTests.pp(y)
    assert(!r.rejectAt(0.05))
  }

  test("PP and ADF agree on stationary series") {
    val y   = stationary(500, phi=0.3)
    val adf = UnitRootTests.adf(y)
    val pp  = UnitRootTests.pp(y)
    assert(adf.rejectAt(0.05) == pp.rejectAt(0.05))
  }

  test("PP toString contains key fields") {
    val y = stationary(200)
    val r = UnitRootTests.pp(y)
    assert(r.toString.contains("bandwidth"))
    assert(r.toString.contains("CV 5%"))
  }

  // --- KPSS ---

  test("KPSS statistic is finite") {
    val y = stationary(200)
    val r = UnitRootTests.kpss(y)
    assert(!r.statistic.isNaN && !r.statistic.isInfinite)
  }

  test("KPSS fails to reject for stationary series") {
    val y = stationary(500, phi=0.3)
    val r = UnitRootTests.kpss(y)
    assert(!r.rejectAt(0.05))
  }

  test("KPSS rejects for random walk") {
    val y = randomWalk(500)
    val r = UnitRootTests.kpss(y)
    assert(r.rejectAt(0.05))
  }

  test("KPSS and ADF give consistent conclusions") {
    // For stationary: ADF rejects, KPSS does not
    val y    = stationary(500, phi=0.3)
    val adf  = UnitRootTests.adf(y)
    val kpss = UnitRootTests.kpss(y)
    assert(adf.rejectAt(0.05) && !kpss.rejectAt(0.05))
  }

  test("KPSS works with trend regime") {
    val y = stationary(200)
    val r = UnitRootTests.kpss(y, regime="trend")
    assert(!r.statistic.isNaN)
  }

  test("KPSS throws on invalid regime") {
    val y = stationary(200)
    assertThrows[IllegalArgumentException] {
      UnitRootTests.kpss(y, regime="drift")
    }
  }

  test("KPSS toString contains key fields") {
    val y = stationary(200)
    val r = UnitRootTests.kpss(y)
    assert(r.toString.contains("statistic"))
    assert(r.toString.contains("bandwidth"))
  }
}