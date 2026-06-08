package series.kalman

import breeze.linalg._

// Carries both diffuse and stationary covariance components
case class DiffuseKalmanState(
                               x: DenseVector[Double],
                               Pinf: DenseMatrix[Double],  // diffuse component
                               Pstar: DenseMatrix[Double]  // stationary component
                             )

object DiffuseKalmanFilter {

  // Threshold for declaring P_inf collapsed to zero
  val threshold: Double = 1e-7

  def isDiffuseComplete(Pinf: DenseMatrix[Double]): Boolean =
    norm(Pinf.toDenseVector) < threshold

  def predict(state: DiffuseKalmanState, model: KalmanModel): DiffuseKalmanState = {
    val x_pred  = model.F * state.x
    val Pi_pred = model.F * state.Pinf * model.F.t
    val Ps_pred = model.F * state.Pstar * model.F.t + model.Q
    DiffuseKalmanState(x_pred, Pi_pred, Ps_pred)
  }

  // Diffuse update equations from Durbin & Koopman (2001)
  def updateDiffuse(
                     state: DiffuseKalmanState,
                     model: KalmanModel,
                     z: DenseVector[Double]
                   ): DiffuseKalmanState = {
    val Finf  = model.H * state.Pinf * model.H.t
    val Fstar = model.H * state.Pstar * model.H.t + model.R

    val y = z - model.H * state.x  // innovation

    if (norm(Finf.toDenseVector) < threshold) {
      // Finf has collapsed — fall through to standard update on Pstar
      val K     = state.Pstar * model.H.t * inv(Fstar)
      val x_upd = state.x + K * y
      val Ps_upd = state.Pstar - K * model.H * state.Pstar
      DiffuseKalmanState(x_upd, DenseMatrix.zeros[Double](state.Pinf.rows, state.Pinf.cols), Ps_upd)
    } else {
      val FinfInv = inv(Finf)
      val Kinf    = state.Pinf * model.H.t * FinfInv
      val Kstar   = state.Pstar * model.H.t * FinfInv -
        state.Pinf * model.H.t * FinfInv * Fstar * FinfInv

      val x_upd   = state.x + Kinf * y
      val Pi_upd  = state.Pinf - Kinf * model.H * state.Pinf
      val Ps_upd  = state.Pstar - Kstar * model.H * state.Pinf -
        Kinf * model.H * state.Pstar
      DiffuseKalmanState(x_upd, Pi_upd, Ps_upd)
    }
  }

  def step(
            state: DiffuseKalmanState,
            model: KalmanModel,
            z: DenseVector[Double]
          ): DiffuseKalmanState = {
    val predicted = predict(state, model)
    if (isDiffuseComplete(predicted.Pinf)) {
      val s = KalmanState(predicted.x, predicted.Pstar)
      val u = KalmanFilter.update(s, model, z)
      DiffuseKalmanState(u.x, DenseMatrix.zeros[Double](predicted.Pinf.rows, predicted.Pinf.cols), u.P)
    } else {
      updateDiffuse(predicted, model, z)
    }
  }

  def run(
           initial: DiffuseKalmanState,
           model: KalmanModel,
           observations: Seq[DenseVector[Double]]
         ): Seq[DiffuseKalmanState] =
    observations.scanLeft(initial)(step(_, model, _))

  // Log likelihood contribution — different formula during diffuse phase
  def logLikelihoodContribution(
                                 state: DiffuseKalmanState,
                                 model: KalmanModel,
                                 z: DenseVector[Double]
                               ): Double = {
    val Finf  = model.H * state.Pinf * model.H.t
    val Fstar = model.H * state.Pstar * model.H.t + model.R
    val y     = z - model.H * state.x

    if (norm(Finf.toDenseVector) < threshold)
    // Standard likelihood contribution
      -0.5 * (math.log(det(Fstar)) + (y.t * inv(Fstar) * y))
    else
    // Diffuse likelihood contribution — Durbin & Koopman eq. 7.2
      -0.5 * (math.log(det(Finf)) + (y.t * inv(Finf) * y))
  }
}