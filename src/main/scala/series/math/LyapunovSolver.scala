package series.math

import breeze.linalg._

object LyapunovSolver {

  /**
   * Solves the discrete Lyapunov equation P = F * P * F.t + Q
   * using the doubling algorithm.
   *
   * Converges quadratically — typically done in 30 iterations or fewer.
   *
   * @param F state transition matrix
   * @param Q process noise covariance
   * @param maxIter maximum iterations
   * @param tol convergence tolerance
   * @return stationary covariance matrix P
   */
  def solve(
             F: DenseMatrix[Double],
             Q: DenseMatrix[Double],
             maxIter: Int = 100,
             tol: Double = 1e-10
           ): DenseMatrix[Double] = {
    var A = F.copy
    var P = Q.copy
    var iter = 0

    while (iter < maxIter) {
      val P_next = A * P * A.t + P
      val A_next = A * A

      val diff = norm((P_next - P).toDenseVector)
      P = P_next
      A = A_next
      iter += 1

      if (diff < tol) return P
    }

    P
  }
}