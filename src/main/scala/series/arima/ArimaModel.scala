package series.arima

import breeze.linalg._
import series.kalman._
import series.math.LyapunovSolver

/**
 * ARIMA(p, d, q) model in state space form.
 *
 * The state space representation follows Durbin & Koopman (2001),
 * Section 3.3. Any ARIMA(p, d, q) model can be written as a linear
 * Gaussian state space model, allowing exact MLE via the Kalman filter.
 *
 * @param p autoregressive order
 * @param d degree of differencing
 * @param q moving average order
 */
case class ArimaModel(p: Int, d: Int, q: Int) {

  // State dimension — companion form
  val stateDim: Int = math.max(p, q + 1) + d

  /**
   * Builds the Kalman model matrices from ARIMA parameters.
   *
   * @param phi   AR coefficients (length p)
   * @param theta MA coefficients (length q)
   * @param sigma2 innovation variance
   * @return KalmanModel with F, H, Q, R matrices
   */
  def toKalmanModel(
                     phi:    Array[Double],
                     theta:  Array[Double],
                     sigma2: Double
                   ): KalmanModel = {
    require(phi.length == p, s"Expected $p AR coefficients, got ${phi.length}")
    require(theta.length == q, s"Expected $q MA coefficients, got ${theta.length}")
    require(sigma2 > 0, "Innovation variance must be positive")

    val m = stateDim

    // Companion form transition matrix F
    val F = DenseMatrix.zeros[Double](m, m)

    // Fill AR coefficients in first row — pad with zeros if p < m
    for (i <- 0 until p) F(0, i) = phi(i)

    // Unit roots from differencing
    for (i <- 0 until d) F(0, p + i) = 1.0

    // Shift operator on subdiagonal
    for (i <- 1 until m) F(i, i - 1) = 1.0

    // Observation matrix H — observes first state element
    val H = DenseMatrix.zeros[Double](1, m)
    H(0, 0) = 1.0

    // Selection vector G for MA component
    val G = DenseVector.zeros[Double](m)
    G(0) = 1.0
    for (j <- 0 until q) G(j + 1) = theta(j)

    // Process noise Q = sigma2 * G * G.t
    val Q = (G * G.t) * sigma2

    // Observation noise R = 0 (innovations absorbed into Q)
    val R = DenseMatrix.zeros[Double](1, 1)

    KalmanModel(F, H, Q, R)
  }

  /**
   * Builds the mixed initial state for the Kalman filter.
   * Stationary component initialized via Lyapunov equation.
   * Nonstationary component (unit roots) initialized diffusely.
   *
   * @param phi    AR coefficients
   * @param theta  MA coefficients
   * @param sigma2 innovation variance
   * @return DiffuseKalmanState with correct Pinf and Pstar
   */
  def initialState(
                    phi:    Array[Double],
                    theta:  Array[Double],
                    sigma2: Double
                  ): DiffuseKalmanState = {
    val model = toKalmanModel(phi, theta, sigma2)
    val m = stateDim

    // Partition state into stationary (AR) and nonstationary (differences)
    val Pinf  = DenseMatrix.zeros[Double](m, m)
    val Pstar = DenseMatrix.zeros[Double](m, m)

    if (d > 0) {
      // Nonstationary states get diffuse initialization
      for (i <- p until p + d) Pinf(i, i) = 1.0
    }

    if (p > 0) {
      // Stationary AR states — solve Lyapunov for Pstar
      val Fp = model.F(0 until p, 0 until p)
      val Qp = model.Q(0 until p, 0 until p)
      val Ps = LyapunovSolver.solve(Fp, Qp)
      Pstar(0 until p, 0 until p) := Ps
    } else {
      // Pure MA or pure difference — diffuse initialization for all
      for (i <- 0 until m) Pinf(i, i) = 1.0
    }

    DiffuseKalmanState(
      x     = DenseVector.zeros[Double](m),
      Pinf  = Pinf,
      Pstar = Pstar
    )
  }
}