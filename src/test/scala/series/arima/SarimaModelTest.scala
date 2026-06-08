package series.arima

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class SarimaModelTest extends AnyFunSuite {

  test("seasonal period must be greater than 1") {
    assertThrows[IllegalArgumentException] {
      SarimaModel(1, 0, 0, 1, 0, 0, s = 1)
    }
  }

  test("SARIMA(1,0,0)(1,0,0)_12 has correct AR order") {
    val model = SarimaModel(p=1, d=0, q=0, P=1, D=0, Q=0, s=12)
    assert(model.arOrder == 13)
  }

  test("SARIMA(0,0,1)(0,0,1)_12 has correct MA order") {
    val model = SarimaModel(p=0, d=0, q=1, P=0, D=0, Q=1, s=12)
    assert(model.maOrder == 13)
  }

  test("AR polynomial expansion is correct for SAR(1)_4") {
    // phi(B) = 1, Phi(B^4): combined should have coeff at lag 4
    val model    = SarimaModel(p=0, d=0, q=0, P=1, D=0, Q=0, s=4)
    val combined = model.expandAR(Array.empty, Array(0.6))
    assert(combined(3) == 0.6)  // lag 4 position
  }

  test("MA polynomial expansion is correct for SMA(1)_4") {
    val model    = SarimaModel(p=0, d=0, q=0, P=0, D=0, Q=1, s=4)
    val combined = model.expandMA(Array.empty, Array(0.5))
    assert(combined(3) == 0.5)  // lag 4 position
  }

  test("cross terms in AR expansion are correct") {
    val model = SarimaModel(p = 1, d = 0, q = 0, P = 1, D = 0, Q = 0, s = 4)
    val combined = model.expandAR(Array(0.3), Array(0.6))
    assert(combined(0) == 0.3)
    assert(combined(3) == 0.6)
    assert(math.abs(combined(4) - (-0.3 * 0.6)) < 1e-10)
  }

  test("state space matrices have correct dimensions") {
    val model = SarimaModel(p=1, d=1, q=1, P=1, D=1, Q=1, s=12)
    val km    = model.toKalmanModel(
      Array(0.3), Array(0.5), Array(0.2), Array(0.4), 1.0
    )
    assert(km.F.rows == model.stateDim)
    assert(km.H.cols == model.stateDim)
    assert(km.Q.rows == model.stateDim)
  }

  test("initial state Pinf is nonzero for seasonal differencing") {
    val model   = SarimaModel(p=1, d=0, q=0, P=0, D=1, Q=0, s=12)
    val initial = model.initialState(Array(0.3), Array.empty, Array.empty, Array.empty, 1.0)
    assert(norm(initial.Pinf.toDenseVector) > 0.0)
  }

  test("throws on wrong coefficient lengths") {
    val model = SarimaModel(p=1, d=0, q=0, P=1, D=0, Q=0, s=12)
    assertThrows[IllegalArgumentException] {
      model.toKalmanModel(Array(0.3, 0.1), Array(0.5), Array.empty, Array.empty, 1.0)
    }
  }
}