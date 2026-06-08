package series.kalman

import breeze.linalg._

object LogLikelihoodAccumulator {

  /**
   * Accumulates the exact log likelihood over a sequence of observations
   * using the diffuse Kalman filter. Handles the diffuse phase and
   * standard phase separately per Durbin & Koopman (2001).
   *
   * @param initial initial diffuse state
   * @param model   Kalman model matrices
   * @param obs     sequence of observations
   * @return exact log likelihood
   */
  def logLikelihood(
                     initial: DiffuseKalmanState,
                     model: KalmanModel,
                     obs: Seq[DenseVector[Double]]
                   ): Double = {
    val n = obs.length
    val k = obs.head.length  // observation dimension

    var state = initial
    var ll = 0.0
    var diffuseCount = 0

    for (z <- obs) {
      val predicted = DiffuseKalmanFilter.predict(state, model)
      ll += DiffuseKalmanFilter.logLikelihoodContribution(predicted, model, z)
      if (!DiffuseKalmanFilter.isDiffuseComplete(predicted.Pinf))
        diffuseCount += 1
      state = DiffuseKalmanFilter.step(state, model, z)
    }

    // Durbin & Koopman diffuse likelihood correction term
    ll - 0.5 * diffuseCount * math.log(2 * math.Pi)
  }

  /**
   * Conditional sum of squares (CSS) approximation.
   * Ignores the diffuse phase entirely — fast but approximate.
   * Useful as a starting point for exact MLE optimization.
   *
   * @param initial initial diffuse state
   * @param model   Kalman model matrices
   * @param obs     sequence of observations
   * @return CSS log likelihood
   */
  def cssLogLikelihood(
                        initial: DiffuseKalmanState,
                        model: KalmanModel,
                        obs: Seq[DenseVector[Double]]
                      ): Double = {
    var state = initial
    var ss = 0.0
    var count = 0

    for (z <- obs) {
      val predicted = DiffuseKalmanFilter.predict(state, model)
      // Only accumulate once diffuse phase has collapsed
      if (DiffuseKalmanFilter.isDiffuseComplete(predicted.Pinf)) {
        val y = z - model.H * predicted.x
        ss += (y.t * y)
        count += 1
      }
      state = DiffuseKalmanFilter.step(state, model, z)
    }

    if (count == 0) Double.NegativeInfinity
    else -0.5 * count * math.log(ss / count)
  }
}