package series.transform

/**
 * Differencing and undifferencing operations for time series.
 */
object Differencing {

  // ── Non-seasonal differencing ──────────────────────────────────────────────

  def diff(y: Array[Double], d: Int = 1): Array[Double] = {
    require(d >= 0, "differencing order must be non-negative")
    require(y.length > d, s"series length ${y.length} must exceed order $d")
    var cur = y
    for (_ <- 0 until d)
      cur = cur.sliding(2).map(w => w(1) - w(0)).toArray
    cur
  }

  /**
   * Inverts d-th order differencing.
   *
   * @param dy       differenced series (length n - d)
   * @param initials the first d values of each intermediate series:
   *                 initials(0) = y(0)
   *                 initials(1) = (diff(y, 1))(0) = y(1) - y(0)  if d >= 2
   *                 etc.
   */
  def undiff(dy: Array[Double], initials: Array[Double], d: Int = 1): Array[Double] = {
    require(initials.length == d, s"need $d initial values, got ${initials.length}")
    // We peel off differencing layers from innermost out.
    // initials(d-1) is the first value of the (d-1)-th differenced series,
    // initials(0) is the first value of the original series.
    var cur = dy
    for (i <- (0 until d).reverse) {
      val init = initials(i)
      val rec  = new Array[Double](cur.length + 1)
      rec(0) = init
      for (t <- cur.indices) rec(t + 1) = rec(t) + cur(t)
      cur = rec
    }
    cur
  }

  // ── Seasonal differencing ──────────────────────────────────────────────────

  def seasonalDiff(y: Array[Double], s: Int, D: Int = 1): Array[Double] = {
    require(s >= 2,  "seasonal period must be at least 2")
    require(D >= 0,  "seasonal differencing order must be non-negative")
    require(y.length > D * s, "series too short for seasonal differencing")
    var cur = y
    for (_ <- 0 until D)
      cur = (s until cur.length).map(t => cur(t) - cur(t - s)).toArray
    cur
  }

  def seasonalUndiff(
    dy:            Array[Double],
    seasonalInits: Array[Double],
    s:             Int,
    D:             Int = 1
  ): Array[Double] = {
    require(seasonalInits.length == s * D,
      s"need ${s*D} seasonal initial values, got ${seasonalInits.length}")
    var cur = dy
    for (pass <- 0 until D) {
      val initStart = s * (D - 1 - pass)
      val inits     = seasonalInits.slice(initStart, initStart + s)
      val rec       = new Array[Double](inits.length + cur.length)
      for (i <- inits.indices) rec(i) = inits(i)
      for (t <- cur.indices)   rec(t + s) = cur(t) + rec(t)
      cur = rec
    }
    cur
  }

  def diffCombined(
    y: Array[Double],
    d: Int,
    D: Int,
    s: Int
  ): (Array[Double], Array[Double], Array[Double]) = {
    val seasonalInits    = y.take(s * D)
    val afterSeasonal    = seasonalDiff(y, s, D)
    val nonSeasonalInits = (0 until d).map(i => diff(afterSeasonal, i)(0)).toArray
    val result           = diff(afterSeasonal, d)
    (result, seasonalInits, nonSeasonalInits)
  }

  def undiffCombined(
    dy:               Array[Double],
    d:                Int,
    D:                Int,
    s:                Int,
    nonSeasonalInits: Array[Double],
    seasonalInits:    Array[Double]
  ): Array[Double] = {
    val afterNonSeasonal = undiff(dy, nonSeasonalInits, d)
    seasonalUndiff(afterNonSeasonal, seasonalInits, s, D)
  }

  // ── Fractional differencing ────────────────────────────────────────────────

  sealed trait FracDiffMethod
  case object MacKinnonDavidson extends FracDiffMethod
  case object Hosking           extends FracDiffMethod

  def fracDiff(
    y:      Array[Double],
    d:      Double,
    L:      Int    = -1,
    method: FracDiffMethod = MacKinnonDavidson
  ): FracDiffResult = {
    require(y.length >= 2, "series must have at least 2 observations")
    val lag = if (L < 0) y.length - 1 else math.min(L, y.length - 1)

    val weights = method match {
      case MacKinnonDavidson => macKinnonDavidsonWeights(d, lag)
      case Hosking           => hoskingWeights(d, lag)
    }

    val result = (lag until y.length).map { t =>
      (0 to math.min(lag, t)).map(k => weights(k) * y(t - k)).sum
    }.toArray

    FracDiffResult(series = result, weights = weights, d = d, L = lag,
      method = method, nDropped = lag)
  }

  private def macKinnonDavidsonWeights(d: Double, L: Int): Array[Double] = {
    val w = new Array[Double](L + 1)
    w(0) = 1.0
    for (k <- 1 to L)
      w(k) = w(k - 1) * (k - 1.0 - d) / k.toDouble
    w
  }

  private def hoskingWeights(d: Double, L: Int): Array[Double] = {
    val w = new Array[Double](L + 1)
    w(0) = 1.0
    for (k <- 1 to L)
      w(k) = w(k - 1) * (k - 1.0 - d) / k.toDouble
    w
  }

  def fracUndiff(
    dy:     Array[Double],
    d:      Double,
    L:      Int    = -1,
    method: FracDiffMethod = MacKinnonDavidson
  ): FracDiffResult = fracDiff(dy, -d, L, method)
}

case class FracDiffResult(
  series:   Array[Double],
  weights:  Array[Double],
  d:        Double,
  L:        Int,
  method:   Differencing.FracDiffMethod,
  nDropped: Int
) {
  override def toString: String = {
    val methodStr = method match {
      case Differencing.MacKinnonDavidson => "MacKinnon-Davidson"
      case Differencing.Hosking           => "Hosking"
    }
    s"FracDiff(d=$d, L=$L, method=$methodStr, " +
    s"seriesLength=${series.length}, dropped=$nDropped)\n" +
    s"  Weights[0..${math.min(5, L)}]: " +
    weights.take(6).map(w => f"$w%.4f").mkString(", ") +
    (if (L > 5) ", ..." else "")
  }
}
