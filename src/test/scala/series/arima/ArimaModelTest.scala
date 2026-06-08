package series.arima

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class ArimaModelTest extends AnyFunSuite {

  test("AR(1) state space has correct dimensions") {
    val model = ArimaModel(p = 1, d = 0, q = 0)
    val km    = model.toKalmanModel(Array(0.5), Array.empty, 1.0)
    assert(km.F.rows == model.stateDim)
    assert(km.F.cols == model.stateDim)
    assert(km.H.rows == 1)
    assert(km.H.cols == model.stateDim)
  }

  test("AR(1) transition matrix has correct structure") {
    val model = ArimaModel(p = 1, d = 0, q = 0)
    val km    = model.toKalmanModel(Array(0.5), Array.empty, 1.0)
    assert(km.F(0, 0) == 0.5)
  }

  test("ARIMA(1,1,0) state dimension is correct") {
    val model = ArimaModel(p = 1, d = 1, q = 0)
    assert(model.stateDim == 2)
  }

  test("ARIMA(1,1,1) state dimension is correct") {
    val model = ArimaModel(p = 1, d = 1, q = 1)
    assert(model.stateDim == 3)
  }

  test("MA(1) state space has correct G vector structure") {
    val model = ArimaModel(p = 0, d = 0, q = 1)
    val km    = model.toKalmanModel(Array.empty, Array(0.5), 1.0)
    // Q should be sigma2 * G * G.t — nonzero in (0,0), (0,1), (1,0), (1,1)
    assert(km.Q(0, 0) > 0.0)
    assert(km.Q(0, 1) != 0.0)
  }

  test("initial state has correct Pinf for ARIMA(1,1,0)") {
    val model   = ArimaModel(p = 1, d = 1, q = 0)
    val initial = model.initialState(Array(0.5), Array.empty, 1.0)
    // Nonstationary state should have diffuse initialization
    assert(initial.Pinf(1, 1) == 1.0)
    // Stationary AR state should have zero Pinf
    assert(initial.Pinf(0, 0) == 0.0)
  }

  test("initial state Pstar is positive for stationary AR component") {
    val model   = ArimaModel(p = 1, d = 0, q = 0)
    val initial = model.initialState(Array(0.5), Array.empty, 1.0)
    assert(initial.Pstar(0, 0) > 0.0)
  }

  test("throws on wrong number of AR coefficients") {
    val model = ArimaModel(p = 2, d = 0, q = 0)
    assertThrows[IllegalArgumentException] {
      model.toKalmanModel(Array(0.5), Array.empty, 1.0)
    }
  }

  test("throws on non-positive sigma2") {
    val model = ArimaModel(p = 1, d = 0, q = 0)
    assertThrows[IllegalArgumentException] {
      model.toKalmanModel(Array(0.5), Array.empty, -1.0)
    }
  }
}