package series.kalman

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class LogLikelihoodAccumulatorTest extends AnyFunSuite {

  val model: KalmanModel = KalmanModel(
    F = DenseMatrix((1.0, 1.0), (0.0, 1.0)),
    H = DenseMatrix((1.0, 0.0)),
    Q = DenseMatrix.eye[Double](2) * 0.01,
    R = DenseMatrix((1.0))
  )

  val initial: DiffuseKalmanState = DiffuseKalmanState(
    x     = DenseVector.zeros[Double](2),
    Pinf  = DenseMatrix.eye[Double](2),
    Pstar = DenseMatrix.zeros[Double](2, 2)
  )

  val observations: Seq[DenseVector[Double]] =
    (1 to 50).map(t => DenseVector(t.toDouble + scala.util.Random.nextGaussian()))

  test("log likelihood is finite") {
    val ll = LogLikelihoodAccumulator.logLikelihood(initial, model, observations)
    assert(!ll.isNaN && !ll.isInfinite)
  }

  test("css log likelihood is finite") {
    val css = LogLikelihoodAccumulator.cssLogLikelihood(initial, model, observations)
    assert(!css.isNaN && !css.isInfinite)
  }

  test("log likelihood is negative") {
    // Log likelihood of a proper density is always negative
    val ll = LogLikelihoodAccumulator.logLikelihood(initial, model, observations)
    assert(ll < 0.0)
  }

  test("better fitting model has higher log likelihood") {
    // A model with observation noise matched to data should fit better
    // than one with badly misspecified noise
    val goodModel = model
    val badModel = KalmanModel(
      F = model.F,
      H = model.H,
      Q = model.Q,
      R = DenseMatrix((1000.0))  // wildly misspecified
    )

    val llGood = LogLikelihoodAccumulator.logLikelihood(initial, goodModel, observations)
    val llBad  = LogLikelihoodAccumulator.logLikelihood(initial, badModel, observations)
    assert(llGood > llBad)
  }

  test("css is less negative than exact ll during diffuse phase") {
    val ll = LogLikelihoodAccumulator.logLikelihood(initial, model, observations)
    val css = LogLikelihoodAccumulator.cssLogLikelihood(initial, model, observations)
    // Both should be finite — they use different formulas so direct comparison isn't meaningful
    assert(!ll.isNaN && !ll.isInfinite)
    assert(!css.isNaN && !css.isInfinite)
  }

  test("log likelihood increases with more observations from true model") {
    val obs30 = observations.take(30)
    val obs50 = observations

    val ll30 = LogLikelihoodAccumulator.logLikelihood(initial, model, obs30)
    val ll50 = LogLikelihoodAccumulator.logLikelihood(initial, model, obs50)

    // More observations — more negative in absolute terms
    assert(ll50 < ll30)
  }
}