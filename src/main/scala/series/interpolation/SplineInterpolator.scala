package series.interpolation

import breeze.linalg._

/**
 * Cubic spline and PCHIP interpolation.
 *
 * Two methods:
 *   Natural cubic spline — C2 continuous, minimizes curvature globally
 *   PCHIP (Fritsch-Carlson) — C1 continuous, preserves monotonicity locally
 *
 * PCHIP is the MATLAB default and preferred for data with local features
 * (peaks, valleys) that cubic splines tend to overshoot.
 *
 * Natural cubic spline is preferred when global smoothness matters more
 * than local shape preservation.
 */
object SplineInterpolator {

  /**
   * Natural cubic spline interpolation.
   *
   * Fits a piecewise cubic polynomial with:
   *   - Exact interpolation at knots
   *   - C2 continuity (continuous second derivatives)
   *   - Natural boundary: second derivative = 0 at endpoints
   *
   * @param xKnown known x coordinates (strictly increasing)
   * @param yKnown known y values
   * @return CubicSpline that can be evaluated at any point
   */
  def naturalCubic(
    xKnown: Array[Double],
    yKnown: Array[Double]
  ): CubicSpline = {
    val n  = xKnown.length
    require(n >= 3, "need at least 3 points for cubic spline")
    require(n == yKnown.length, "x and y must match")

    val h = (0 until n-1).map(i => xKnown(i+1) - xKnown(i)).toArray

    // Set up tridiagonal system for second derivatives (sigma)
    // Natural boundary: sigma(0) = sigma(n-1) = 0
    val m  = n - 2
    val rhs = Array.tabulate(m) { i =>
      6.0 * ((yKnown(i+2) - yKnown(i+1)) / h(i+1) -
             (yKnown(i+1) - yKnown(i))   / h(i))
    }

    val mainDiag = Array.tabulate(m)(i => 2.0 * (h(i) + h(i+1)))
    val offDiag  = Array.tabulate(m-1)(i => h(i+1))

    // Solve tridiagonal system (Thomas algorithm)
    val sigma = new Array[Double](n)  // natural BCs: sigma(0) = sigma(n-1) = 0
    if (m > 0) {
      val s = solveTridiagonal(mainDiag, offDiag, offDiag, rhs)
      for (i <- 0 until m) sigma(i+1) = s(i)
    }

    // Compute cubic coefficients for each interval [x_i, x_{i+1}]
    val coeffs = Array.tabulate(n-1) { i =>
      val hi = h(i)
      val a  = yKnown(i)
      val b  = (yKnown(i+1) - yKnown(i)) / hi -
               hi * (2*sigma(i) + sigma(i+1)) / 6.0
      val c  = sigma(i) / 2.0
      val d  = (sigma(i+1) - sigma(i)) / (6.0 * hi)
      Array(a, b, c, d)
    }

    CubicSpline(xKnown, yKnown, coeffs, SplineType.Natural)
  }

  /**
   * PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
   * using the Fritsch-Carlson method.
   *
   * Computes derivatives at each knot that preserve local monotonicity.
   * This is what MATLAB's pchip() and interp1(..., 'pchip') use.
   *
   * @param xKnown known x coordinates (strictly increasing)
   * @param yKnown known y values
   * @return CubicSpline with PCHIP coefficients
   */
  def pchip(
    xKnown: Array[Double],
    yKnown: Array[Double]
  ): CubicSpline = {
    val n = xKnown.length
    require(n >= 2, "need at least 2 points for PCHIP")
    require(n == yKnown.length, "x and y must match")

    val h   = (0 until n-1).map(i => xKnown(i+1) - xKnown(i)).toArray
    val del = (0 until n-1).map(i => (yKnown(i+1) - yKnown(i)) / h(i)).toArray

    // Fritsch-Carlson derivatives
    val d = fritschCarlsonDerivatives(h, del, n)

    // Hermite cubic coefficients for each interval
    val coeffs = Array.tabulate(n-1) { i =>
      val hi  = h(i)
      val dy  = yKnown(i+1) - yKnown(i)
      val a   = yKnown(i)
      val b   = d(i)
      val c   = (3*del(i) - 2*d(i) - d(i+1)) / hi
      val dd  = (d(i) + d(i+1) - 2*del(i)) / (hi * hi)
      Array(a, b, c, dd)
    }

    CubicSpline(xKnown, yKnown, coeffs, SplineType.PCHIP)
  }

  /**
   * Fritsch-Carlson algorithm for monotonicity-preserving derivatives.
   */
  private def fritschCarlsonDerivatives(
    h:   Array[Double],
    del: Array[Double],
    n:   Int
  ): Array[Double] = {
    val d = new Array[Double](n)

    // Endpoint derivatives (non-centered finite difference)
    d(0)   = del(0)
    d(n-1) = del(n-2)

    // Interior: harmonic mean of adjacent slopes (when same sign)
    for (i <- 1 until n-1) {
      if (del(i-1) * del(i) <= 0.0) {
        // Sign change or zero: set derivative to zero (local extremum)
        d(i) = 0.0
      } else {
        // Harmonic mean weighted by interval lengths
        val w1 = 2*h(i)   + h(i-1)
        val w2 = h(i)   + 2*h(i-1)
        d(i) = (w1 + w2) / (w1 / del(i-1) + w2 / del(i))
      }
    }

    // Fritsch-Carlson monotonicity conditions
    for (i <- 0 until n-1) {
      if (math.abs(del(i)) < 1e-10) {
        d(i) = 0.0; d(i+1) = 0.0
      } else {
        val alpha = d(i) / del(i)
        val beta  = d(i+1) / del(i)
        val rho   = math.sqrt(alpha*alpha + beta*beta)
        if (rho > 3.0) {
          val tau = 3.0 / rho
          d(i)   = tau * alpha * del(i)
          d(i+1) = tau * beta  * del(i)
        }
      }
    }

    d
  }

  /**
   * Thomas algorithm for tridiagonal systems.
   * Solves A*x = rhs where A is tridiagonal with
   * lower offDiag = lo, main diag = main, upper offDiag = up.
   */
  private def solveTridiagonal(
    main: Array[Double],
    up:   Array[Double],
    lo:   Array[Double],
    rhs:  Array[Double]
  ): Array[Double] = {
    val n   = main.length
    val c   = new Array[Double](n)
    val d   = new Array[Double](n)
    val x   = new Array[Double](n)

    c(0) = up(0) / main(0)
    d(0) = rhs(0) / main(0)

    for (i <- 1 until n) {
      val m = main(i) - lo(i-1) * c(i-1)
      c(i) = if (i < n-1) up(i) / m else 0.0
      d(i) = (rhs(i) - lo(i-1) * d(i-1)) / m
    }

    x(n-1) = d(n-1)
    for (i <- (0 until n-1).reverse)
      x(i) = d(i) - c(i) * x(i+1)

    x
  }
}

object SplineType extends Enumeration {
  val Natural, PCHIP = Value
}

/**
 * Cubic spline that can be evaluated at arbitrary points.
 *
 * @param xKnown  knot x coordinates
 * @param yKnown  knot y values
 * @param coeffs  cubic coefficients for each interval (a, b, c, d)
 * @param splineType type of spline
 */
case class CubicSpline(
  xKnown:     Array[Double],
  yKnown:     Array[Double],
  coeffs:     Array[Array[Double]],
  splineType: SplineType.Value
) {
  /**
   * Evaluates the spline at query points.
   *
   * @param xQuery  query x coordinates
   * @param extrapolate if true use boundary polynomial; else NaN
   * @return interpolated values
   */
  def evaluate(
    xQuery:     Array[Double],
    extrapolate: Boolean = false
  ): Array[Double] = {
    xQuery.map { xq =>
      if (!extrapolate && (xq < xKnown.head || xq > xKnown.last))
        Double.NaN
      else {
        val i   = findInterval(xq)
        val dx  = xq - xKnown(i)
        val c   = coeffs(i)
        c(0) + dx * (c(1) + dx * (c(2) + dx * c(3)))
      }
    }
  }

  private def findInterval(xq: Double): Int = {
    val clipped = math.max(xKnown.head, math.min(xKnown.last, xq))
    var lo = 0; var hi = xKnown.length - 2
    while (lo < hi) {
      val mid = (lo + hi + 1) / 2
      if (xKnown(mid) <= clipped) lo = mid else hi = mid - 1
    }
    lo
  }

  override def toString: String =
    s"CubicSpline($splineType, ${xKnown.length} knots, " +
    s"range=[${xKnown.head}, ${xKnown.last}])"
}
