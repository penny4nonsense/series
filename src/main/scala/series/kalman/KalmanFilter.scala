package series.kalman

import breeze.linalg._

case class KalmanState(x: DenseVector[Double], P: DenseMatrix[Double])

case class KalmanModel(
                        F: DenseMatrix[Double],
                        H: DenseMatrix[Double],
                        Q: DenseMatrix[Double],
                        R: DenseMatrix[Double]
                      )

object KalmanFilter {

  def predict(state: KalmanState, model: KalmanModel): KalmanState = {
    val x_pred = model.F * state.x
    val P_pred = model.F * state.P * model.F.t + model.Q
    KalmanState(x_pred, P_pred)
  }

  def update(state: KalmanState, model: KalmanModel, z: DenseVector[Double]): KalmanState = {
    val S = model.H * state.P * model.H.t + model.R
    val K = state.P * model.H.t * inv(S)
    val y = z - model.H * state.x
    val x_upd = state.x + K * y
    val P_upd = state.P - K * model.H * state.P
    KalmanState(x_upd, P_upd)
  }

  def step(state: KalmanState, model: KalmanModel, z: DenseVector[Double]): KalmanState =
    update(predict(state, model), model, z)

  def run(
           initial: KalmanState,
           model: KalmanModel,
           observations: Seq[DenseVector[Double]]
         ): Seq[KalmanState] =
    observations.scanLeft(initial)(step(_, model, _))
}