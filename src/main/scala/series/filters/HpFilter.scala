package series.filters

import breeze.linalg._

/**
 * Hodrick-Prescott filter for trend-cycle decomposition.
 *
 * Solves the penalized least squares problem:
 *   min_tau sum_t (y_t - tau_t)^2 + lambda * sum_t (Delta^2 tau_t)^2
 *
 * The closed-form solution is:
 *   tau = (I + lambda * D'D)^{-1} y
 *
 * where D is the (n-2) x n second-difference matrix.
 * This is a sparse symmetric positive definite system solved efficiently
 * via the banded structure.
 *
 * Standard lambda values:
 *   1600   — quarterly data (Hodrick-Prescott original)
 *   14400  — monthly data
 *   100    — annual data
 *   6.25   — annual data (Ravn-Uhlig)
 *
 * @param lambda smoothing parameter (larger = smoother trend)
 */
class HpFilter(lambda: Double = 1600.0) {
  require(lambda > 0, "lambda must be positive")

  /**
   * Applies the HP filter to a series.
   *
   * @param y input series
   * @return HpFilterResult with trend and cycle components
   */
  def filter(y: Array[Double]): HpFilterResult = {
    val n  = y.length
    require(n >= 4, "series must have at least 4 observations")

    val yVec = DenseVector(y)

    // Build the penalty matrix lambda * D'D as a full symmetric matrix
    // D'D is a banded symmetric matrix with known structure:
    //   diag: [1, 5, 6, ..., 6, 5, 1] (with modifications at boundaries)
    //   off-diag-1: [-2, -4, ..., -4, -2]
    //   off-diag-2: [1, 1, ..., 1]
    val A = DenseMatrix.zeros[Double](n, n)

    // Build I + lambda * D'D directly
    // D'D has the following structure for rows i (0-indexed):
    for (i <- 0 until n) A(i, i) = 1.0

    // Second difference penalty contributions
    for (i <- 0 until n - 2) {
      A(i,   i)   += lambda
      A(i+1, i+1) += 4.0 * lambda
      A(i+2, i+2) += lambda
      A(i,   i+2) += lambda
      A(i+2, i)   += lambda
      A(i,   i+1) -= 2.0 * lambda
      A(i+1, i)   -= 2.0 * lambda
      A(i+1, i+2) -= 2.0 * lambda
      A(i+2, i+1) -= 2.0 * lambda
    }

    val trend  = inv(A) * yVec
    val cycle  = yVec - trend

    HpFilterResult(
      trend  = trend.toArray,
      cycle  = cycle.toArray,
      lambda = lambda,
      n      = n
    )
  }
}

/**
 * HP filter result.
 *
 * @param trend  smoothed trend component
 * @param cycle  cyclical component (y - trend)
 * @param lambda smoothing parameter used
 * @param n      series length
 */
case class HpFilterResult(
  trend:  Array[Double],
  cycle:  Array[Double],
  lambda: Double,
  n:      Int
) {
  def varianceRatio: Double = {
    val varCycle = cycle.map(c => c*c).sum / n
    val varTrend = {
      val mu = trend.sum / n
      trend.map(t => math.pow(t - mu, 2)).sum / n
    }
    varCycle / math.max(varTrend, 1e-10)
  }

  override def toString: String = {
    val cycMean = cycle.sum / n
    val cycStd  = math.sqrt(cycle.map(c => (c - cycMean)*(c - cycMean)).sum / n)
    f"HP Filter (lambda=$lambda%.1f, n=$n)\n" +
    f"  Trend range: [${trend.min%.4f}, ${trend.max%.4f}]\n" +
    f"  Cycle mean:  $cycMean%.4f\n" +
    f"  Cycle std:   $cycStd%.4f\n"
  }
}
