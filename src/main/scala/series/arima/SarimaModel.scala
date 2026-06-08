package series.arima

import breeze.linalg._
import series.kalman._
import series.math.LyapunovSolver

/**
 * SARIMA(p,d,q)(P,D,Q)_s model in state space form.
 *
 * The seasonal and nonseasonal polynomials are multiplied together
 * to produce the combined AR and MA polynomials, which are then
 * placed into companion form exactly as in ArimaModel.
 *
 * @param p nonseasonal AR order
 * @param d nonseasonal differencing
 * @param q nonseasonal MA order
 * @param P seasonal AR order
 * @param D seasonal differencing
 * @param Q seasonal MA order
 * @param s seasonal period
 */
case class SarimaModel(
                        p: Int, d: Int, q: Int,
                        P: Int, D: Int, Q: Int,
                        s: Int
                      ) {
  require(s > 1, "Seasonal period must be greater than 1")

  // Combined AR order after multiplying polynomials
  val arOrder: Int = p + P * s

  // Combined MA order after multiplying polynomials
  val maOrder: Int = q + Q * s

  // State dimension
  val stateDim: Int = math.max(arOrder, maOrder + 1) + d + D * s

  /**
   * Multiplies nonseasonal and seasonal AR polynomials.
   * phi(B) * Phi(B^s) expanded into combined coefficients.
   *
   * @param phi nonseasonal AR coefficients
   * @param Phi seasonal AR coefficients
   * @return combined AR polynomial coefficients
   */
  def expandAR(phi: Array[Double], Phi: Array[Double]): Array[Double] = {
    val combined = Array.fill(arOrder)(0.0)

    // Nonseasonal terms
    for (i <- phi.indices) combined(i) += phi(i)

    // Seasonal terms
    for (j <- Phi.indices) combined((j + 1) * s - 1) += Phi(j)

    // Cross terms phi * Phi
    for (i <- phi.indices; j <- Phi.indices) {
      val idx = (i + 1) + (j + 1) * s - 1
      if (idx < combined.length)
        combined(idx) -= phi(i) * Phi(j)
    }

    combined
  }

  /**
   * Multiplies nonseasonal and seasonal MA polynomials.
   * theta(B) * Theta(B^s) expanded into combined coefficients.
   *
   * @param theta nonseasonal MA coefficients
   * @param Theta seasonal MA coefficients
   * @return combined MA polynomial coefficients
   */
  def expandMA(theta: Array[Double], Theta: Array[Double]): Array[Double] = {
    val combined = Array.fill(maOrder)(0.0)

    // Nonseasonal terms
    for (i <- theta.indices) combined(i) += theta(i)

    // Seasonal terms
    for (j <- Theta.indices) combined((j + 1) * s - 1) += Theta(j)

    // Cross terms theta * Theta
    for (i <- theta.indices; j <- Theta.indices)
      combined(i + (j + 1) * s - 1) += theta(i) * Theta(j)

    combined
  }

  /**
   * Builds KalmanModel from SARIMA parameters.
   *
   * @param phi    nonseasonal AR coefficients (length p)
   * @param Phi    seasonal AR coefficients (length P)
   * @param theta  nonseasonal MA coefficients (length q)
   * @param Theta  seasonal MA coefficients (length Q)
   * @param sigma2 innovation variance
   * @return KalmanModel with F, H, Q, R matrices
   */
  def toKalmanModel(
                     phi:    Array[Double],
                     Phi:    Array[Double],
                     theta:  Array[Double],
                     Theta:  Array[Double],
                     sigma2: Double
                   ): KalmanModel = {
    require(phi.length == p,   s"Expected $p AR coefficients")
    require(Phi.length == P,   s"Expected $P seasonal AR coefficients")
    require(theta.length == q, s"Expected $q MA coefficients")
    require(Theta.length == Q, s"Expected $Q seasonal MA coefficients")
    require(sigma2 > 0,        "Innovation variance must be positive")

    val combinedAR = expandAR(phi, Phi)
    val combinedMA = expandMA(theta, Theta)
    val m          = stateDim

    // Companion form transition matrix
    val F = DenseMatrix.zeros[Double](m, m)

    // Combined AR coefficients in first row
    for (i <- combinedAR.indices) F(0, i) = combinedAR(i)

    // Unit roots — nonseasonal and seasonal differencing
    for (i <- 0 until d)     F(0, arOrder + i) = 1.0
    for (i <- 0 until D * s) F(0, arOrder + d + i) = 1.0

    // Shift operator on subdiagonal
    for (i <- 1 until m) F(i, i - 1) = 1.0

    // Observation matrix
    val H = DenseMatrix.zeros[Double](1, m)
    H(0, 0) = 1.0

    // Selection vector for combined MA
    val G = DenseVector.zeros[Double](m)
    G(0) = 1.0
    for (j <- combinedMA.indices) G(j + 1) = combinedMA(j)

    // Process noise
    val Qmat = (G * G.t) * sigma2

    // Zero observation noise — innovations form
    val R = DenseMatrix.zeros[Double](1, 1)

    KalmanModel(F, H, Qmat, R)
  }

  /**
   * Mixed initial state — stationary component via Lyapunov,
   * nonstationary (unit roots) initialized diffusely.
   */
  def initialState(
                    phi:    Array[Double],
                    Phi:    Array[Double],
                    theta:  Array[Double],
                    Theta:  Array[Double],
                    sigma2: Double
                  ): DiffuseKalmanState = {
    val km = toKalmanModel(phi, Phi, theta, Theta, sigma2)
    val m  = stateDim

    val Pinf  = DenseMatrix.zeros[Double](m, m)
    val Pstar = DenseMatrix.zeros[Double](m, m)

    val nonstationaryDim = d + D * s

    // Nonstationary states — diffuse
    for (i <- arOrder until arOrder + nonstationaryDim)
      Pinf(i, i) = 1.0

    // Stationary AR states — Lyapunov
    if (arOrder > 0) {
      val Fp = km.F(0 until arOrder, 0 until arOrder)
      val Qp = km.Q(0 until arOrder, 0 until arOrder)
      val Ps = LyapunovSolver.solve(Fp, Qp)
      Pstar(0 until arOrder, 0 until arOrder) := Ps
    } else {
      for (i <- 0 until m) Pinf(i, i) = 1.0
    }

    DiffuseKalmanState(
      x     = DenseVector.zeros[Double](m),
      Pinf  = Pinf,
      Pstar = Pstar
    )
  }
}