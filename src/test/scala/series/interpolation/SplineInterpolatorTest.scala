package series.interpolation

import org.scalatest.funsuite.AnyFunSuite

class SplineInterpolatorTest extends AnyFunSuite {

  val xKnown = Array(0.0, 1.0, 2.0, 3.0, 4.0)
  val yKnown = Array(0.0, 1.0, 4.0, 9.0, 16.0)  // y = x^2

  test("natural cubic interpolates at knots exactly") {
    val spline = SplineInterpolator.naturalCubic(xKnown, yKnown)
    val yHat   = spline.evaluate(xKnown)
    xKnown.zip(yHat).zip(yKnown).foreach { case ((x, yh), y) =>
      assert(math.abs(yh - y) < 1e-8, s"at x=$x: got $yh, expected $y")
    }
  }

  test("pchip interpolates at knots exactly") {
    val spline = SplineInterpolator.pchip(xKnown, yKnown)
    val yHat   = spline.evaluate(xKnown)
    xKnown.zip(yHat).zip(yKnown).foreach { case ((x, yh), y) =>
      assert(math.abs(yh - y) < 1e-8)
    }
  }

  test("natural cubic is smooth between knots") {
    val spline = SplineInterpolator.naturalCubic(xKnown, yKnown)
    val xQ     = Array.tabulate(100)(i => i * 4.0 / 99)
    val yQ     = spline.evaluate(xQ)
    assert(yQ.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("pchip preserves monotonicity") {
    // Monotone increasing data
    val xM = Array(0.0, 1.0, 2.0, 3.0, 4.0)
    val yM = Array(0.0, 0.5, 1.0, 1.5, 2.0)
    val spline = SplineInterpolator.pchip(xM, yM)
    val xQ     = Array.tabulate(50)(i => i * 4.0 / 49)
    val yQ     = spline.evaluate(xQ)
    yQ.sliding(2).foreach { w => assert(w(1) >= w(0) - 1e-10) }
  }

  test("returns NaN outside range when not extrapolating") {
    val spline = SplineInterpolator.naturalCubic(xKnown, yKnown)
    val y = spline.evaluate(Array(-1.0, 5.0), extrapolate = false)
    assert(y.forall(_.isNaN))
  }

  test("extrapolates when requested") {
    val spline = SplineInterpolator.pchip(xKnown, yKnown)
    val y = spline.evaluate(Array(5.0), extrapolate = true)
    assert(!y(0).isNaN)
  }

  test("toString contains type") {
    val spline = SplineInterpolator.pchip(xKnown, yKnown)
    assert(spline.toString.contains("PCHIP"))
  }
}
