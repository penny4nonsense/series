package series.interpolation

/**
 * Linear interpolation for irregularly-spaced time series.
 *
 * Handles:
 *   - Regular grids with missing values (NaN)
 *   - Irregular grids (arbitrary x coordinates)
 *   - Extrapolation (optional)
 */
object LinearInterpolator {

  /**
   * Interpolates missing values (NaN) in a regularly-spaced series.
   * Uses piecewise linear interpolation between known values.
   * Does not extrapolate beyond known endpoints.
   *
   * @param y series with NaN at missing positions
   * @return interpolated series (NaN at leading/trailing missing values)
   */
  def interpolateMissing(y: Array[Double]): Array[Double] = {
    val result = y.clone()
    val n      = y.length

    var i = 0
    while (i < n) {
      if (result(i).isNaN) {
        // Find the surrounding known values
        val left  = (i - 1 to 0 by -1).find(j => !result(j).isNaN)
        val right = (i + 1 until n).find(j => !result(j).isNaN)

        (left, right) match {
          case (Some(l), Some(r)) =>
            // Linear interpolation between l and r
            val slope = (result(r) - result(l)) / (r - l).toDouble
            for (k <- l + 1 until r)
              result(k) = result(l) + slope * (k - l)
            i = r
          case _ =>
            i += 1
        }
      } else i += 1
    }
    result
  }

  /**
   * Interpolates at arbitrary query points given known (x, y) pairs.
   *
   * @param xKnown  known x coordinates (must be strictly increasing)
   * @param yKnown  known y values
   * @param xQuery  query x coordinates
   * @param extrapolate if true, extrapolates beyond endpoints; else NaN
   * @return interpolated y values at xQuery
   */
  def interpolate(
    xKnown:      Array[Double],
    yKnown:      Array[Double],
    xQuery:      Array[Double],
    extrapolate: Boolean = false
  ): Array[Double] = {
    require(xKnown.length == yKnown.length, "x and y must have same length")
    require(xKnown.length >= 2, "need at least 2 known points")
    require(xKnown.sliding(2).forall(w => w(1) > w(0)),
      "x coordinates must be strictly increasing")

    xQuery.map { xq =>
      if (!extrapolate && (xq < xKnown.head || xq > xKnown.last)) {
        Double.NaN
      } else {
        // Find bracketing interval via binary search
        val idx = binarySearch(xKnown, xq)
        val i   = math.max(0, math.min(idx - 1, xKnown.length - 2))
        val x0  = xKnown(i);     val x1 = xKnown(i + 1)
        val y0  = yKnown(i);     val y1 = yKnown(i + 1)
        val t   = (xq - x0) / (x1 - x0)
        y0 + t * (y1 - y0)
      }
    }
  }

  private def binarySearch(x: Array[Double], target: Double): Int = {
    var lo = 0; var hi = x.length - 1
    while (lo < hi) {
      val mid = (lo + hi) / 2
      if (x(mid) < target) lo = mid + 1 else hi = mid
    }
    lo
  }
}
