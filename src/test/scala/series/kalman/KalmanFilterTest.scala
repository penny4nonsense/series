package series.kalman

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class KalmanFilterTest extends AnyFunSuite {

  // Constant velocity model — position and velocity
  val model: KalmanModel = KalmanModel(
    F = DenseMatrix((1.0, 1.0), (0.0, 1.0)),
    H = DenseMatrix((1.0, 0.0)),
    Q = DenseMatrix.eye[Double](2) * 0.01,
    R = DenseMatrix((1.0))
  )

  val initial: KalmanState = KalmanState(
    x = DenseVector(0.0, 1.0),
    P = DenseMatrix.eye[Double](2)
  )

  test("predict increases uncertainty") {
    val predicted = KalmanFilter.predict(initial, model)
    // Trace of P should increase after prediction
    assert(trace(predicted.P) > trace(initial.P))
  }

  test("update reduces uncertainty") {
    val predicted = KalmanFilter.predict(initial, model)
    val updated = KalmanFilter.update(predicted, model, DenseVector(1.0))
    // Trace of P should decrease after update
    assert(trace(updated.P) < trace(predicted.P))
  }

  test("filter converges toward true signal") {
    // True position increases by 1 each step
    val observations = (1 to 50).map(t => DenseVector(t.toDouble))
    val states = KalmanFilter.run(initial, model, observations)

    // After 50 steps, estimated position should be close to true position
    val lastState = states.last
    val lastObs = 50.0
    assert(math.abs(lastState.x(0) - lastObs) < 1.0)
  }

  test("run returns correct number of states") {
    val observations = (1 to 10).map(t => DenseVector(t.toDouble))
    val states = KalmanFilter.run(initial, model, observations)
    // scanLeft returns n+1 elements (initial + one per observation)
    assert(states.length == 11)
  }
}