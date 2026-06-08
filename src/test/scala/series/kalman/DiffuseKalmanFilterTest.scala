package series.kalman

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class DiffuseKalmanFilterTest extends AnyFunSuite {

  val model: KalmanModel = KalmanModel(
    F = DenseMatrix((1.0, 1.0), (0.0, 1.0)),
    H = DenseMatrix((1.0, 0.0)),
    Q = DenseMatrix.eye[Double](2) * 0.01,
    R = DenseMatrix((1.0))
  )

  // Diffuse initial state — large Pinf signals complete ignorance
  val initial: DiffuseKalmanState = DiffuseKalmanState(
    x     = DenseVector.zeros[Double](2),
    Pinf  = DenseMatrix.eye[Double](2),
    Pstar = DenseMatrix.zeros[Double](2, 2)
  )

  test("predict propagates both Pinf and Pstar") {
    val predicted = DiffuseKalmanFilter.predict(initial, model)
    // Pinf should propagate through F
    assert(norm(predicted.Pinf.toDenseVector) > 0.0)
    // Pstar should pick up process noise Q
    assert(norm(predicted.Pstar.toDenseVector) > 0.0)
  }

  test("Pinf collapses after sufficient observations") {
    val observations = (1 to 20).map(t => DenseVector(t.toDouble))
    val states = DiffuseKalmanFilter.run(initial, model, observations)
    val lastState = states.last
    assert(DiffuseKalmanFilter.isDiffuseComplete(lastState.Pinf))
  }

  test("Pstar is positive after diffuse phase collapses") {
    val observations = (1 to 20).map(t => DenseVector(t.toDouble))
    val states = DiffuseKalmanFilter.run(initial, model, observations)
    // Find first collapsed state
    val collapsed = states.find(s => DiffuseKalmanFilter.isDiffuseComplete(s.Pinf))
    assert(collapsed.isDefined)
    assert(trace(collapsed.get.Pstar) > 0.0)
  }

  test("filter converges toward true signal after diffuse phase") {
    val observations = (1 to 50).map(t => DenseVector(t.toDouble))
    val states = DiffuseKalmanFilter.run(initial, model, observations)
    val lastState = states.last
    assert(math.abs(lastState.x(0) - 50.0) < 1.0)
  }

  test("log likelihood contribution is finite after collapse") {
    val observations = (1 to 20).map(t => DenseVector(t.toDouble))
    val states = DiffuseKalmanFilter.run(initial, model, observations)
    // Take a state well past collapse and check likelihood is finite
    val lateState = states(15)
    val ll = DiffuseKalmanFilter.logLikelihoodContribution(
      lateState, model, DenseVector(15.0)
    )
    assert(!ll.isNaN && !ll.isInfinite)
  }

  test("collapsed state matches standard Kalman filter") {
    // Once diffuse phase is done, results should match standard filter
    val observations = (1 to 30).map(t => DenseVector(t.toDouble))
    val diffuseStates = DiffuseKalmanFilter.run(initial, model, observations)

    // Find collapse point
    val collapseIdx = diffuseStates.indexWhere(s =>
      DiffuseKalmanFilter.isDiffuseComplete(s.Pinf)
    )
    assert(collapseIdx > 0)

    // From collapse point, standard and diffuse filters should agree
    val collapseState = diffuseStates(collapseIdx)
    val standardInitial = KalmanState(collapseState.x, collapseState.Pstar)
    val remainingObs = observations.drop(collapseIdx)

    val standardStates = KalmanFilter.run(standardInitial, model, remainingObs)
    val diffuseTail = diffuseStates.drop(collapseIdx)

    standardStates.zip(diffuseTail).foreach { case (s, d) =>
      assert(math.abs(s.x(0) - d.x(0)) < 1e-6)
    }
  }
}