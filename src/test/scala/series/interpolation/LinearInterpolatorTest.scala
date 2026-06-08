package series.interpolation

import org.scalatest.funsuite.AnyFunSuite

class LinearInterpolatorTest extends AnyFunSuite {

  test("interpolates missing values") {
    val y    = Array(1.0, Double.NaN, 3.0)
    val yHat = LinearInterpolator.interpolateMissing(y)
    assert(math.abs(yHat(1) - 2.0) < 1e-10)
  }

  test("does not modify known values") {
    val y    = Array(1.0, Double.NaN, 3.0, Double.NaN, 5.0)
    val yHat = LinearInterpolator.interpolateMissing(y)
    assert(math.abs(yHat(0) - 1.0) < 1e-10)
    assert(math.abs(yHat(2) - 3.0) < 1e-10)
    assert(math.abs(yHat(4) - 5.0) < 1e-10)
  }

  test("handles multiple consecutive missing values") {
    val y    = Array(0.0, Double.NaN, Double.NaN, Double.NaN, 4.0)
    val yHat = LinearInterpolator.interpolateMissing(y)
    assert(math.abs(yHat(1) - 1.0) < 1e-10)
    assert(math.abs(yHat(2) - 2.0) < 1e-10)
    assert(math.abs(yHat(3) - 3.0) < 1e-10)
  }

  test("interpolate at query points") {
    val xK = Array(0.0, 1.0, 2.0)
    val yK = Array(0.0, 2.0, 4.0)
    val yQ = LinearInterpolator.interpolate(xK, yK, Array(0.5, 1.5))
    assert(math.abs(yQ(0) - 1.0) < 1e-10)
    assert(math.abs(yQ(1) - 3.0) < 1e-10)
  }

  test("returns NaN outside range without extrapolation") {
    val xK = Array(0.0, 1.0, 2.0)
    val yK = Array(0.0, 1.0, 2.0)
    val yQ = LinearInterpolator.interpolate(xK, yK, Array(-1.0, 3.0))
    assert(yQ.forall(_.isNaN))
  }

  test("extrapolates when requested") {
    val xK = Array(0.0, 1.0, 2.0)
    val yK = Array(0.0, 1.0, 2.0)
    val yQ = LinearInterpolator.interpolate(xK, yK, Array(3.0), extrapolate = true)
    assert(!yQ(0).isNaN)
  }
}
