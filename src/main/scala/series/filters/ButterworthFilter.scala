package series.filters

/**
 * Butterworth IIR filter with zero-phase forward-backward filtering.
 *
 * Design procedure:
 *   1. Compute analog Butterworth poles
 *   2. Apply bilinear transform with frequency prewarping
 *   3. Convert to digital filter coefficients (b, a polynomials)
 *   4. Apply forward-backward filtering (filtfilt) for zero phase
 *
 * The zero-phase filter doubles the effective order: a user-specified
 * order-n filter has effective order 2n in the output.
 *
 * Filter types: LowPass, HighPass, BandPass
 *
 * @param order     one-sided filter order (effective order = 2*order)
 * @param filterType filter type
 * @param cutoff    normalized cutoff frequency in (0, 1) for LP/HP,
 *                  or (low, high) for BP where 1.0 = Nyquist
 */
class ButterworthFilter(
  order:      Int,
  filterType: ButterworthFilter.FilterType,
  cutoff:     Either[Double, (Double, Double)]
) {
  import ButterworthFilter._

  require(order >= 1, "filter order must be at least 1")

  cutoff match {
    case Left(f)       => require(f > 0 && f < 1, "cutoff must be in (0,1)")
    case Right((l, h)) =>
      require(l > 0 && l < 1, "low cutoff must be in (0,1)")
      require(h > l && h < 1, "high cutoff must exceed low cutoff and be < 1")
  }

  filterType match {
    case BandPass => require(cutoff.isRight, "BandPass requires two cutoff frequencies")
    case _        => require(cutoff.isLeft,  "LowPass/HighPass require one cutoff frequency")
  }

  /**
   * Prewarp frequency from digital (normalized, 0..1 = 0..Nyquist)
   * to analog domain using bilinear transform prewarping.
   * omega_analog = 2 * tan(pi * f)
   */
  private def prewarp(f: Double): Double = 2.0 * math.tan(math.Pi * f)

  /**
   * Computes analog Butterworth poles for a normalized low-pass filter.
   * Poles lie on the unit circle in the left half-plane:
   *   p_k = exp(i * pi * (2k + n - 1) / (2n))  for k = 1..n
   */
  private def analogPoles(n: Int): Array[(Double, Double)] = {
    (1 to n).map { k =>
      val angle = math.Pi * (2 * k + n - 1).toDouble / (2 * n)
      (math.cos(angle), math.sin(angle))
    }.toArray
  }

  /**
   * Applies the bilinear transform to map analog poles/zeros to digital.
   * z = (2 + s) / (2 - s) where s is the analog frequency variable.
   * For a pole at s = p: z = (2 + p) / (2 - p)
   */
  private def bilinear(re: Double, im: Double): (Double, Double) = {
    // z = (2 + p) / (2 - p)
    val numRe = 2.0 + re
    val numIm = im
    val denRe = 2.0 - re
    val denIm = -im
    val denom = denRe * denRe + denIm * denIm
    ((numRe * denRe + numIm * denIm) / denom,
     (numIm * denRe - numRe * denIm) / denom)
  }

  /**
   * Designs the digital filter coefficients (b, a) for a low-pass filter.
   * Returns (b coefficients, a coefficients) where:
   *   H(z) = B(z) / A(z)
   *   B(z) = b(0) + b(1)*z^{-1} + ...
   *   A(z) = 1    + a(1)*z^{-1} + ...
   */
  private def designLP(omegaC: Double): (Array[Double], Array[Double]) = {
    val n      = order
    val poles  = analogPoles(n)

    // Scale poles to cutoff frequency
    val scaledPoles = poles.map { case (re, im) => (re * omegaC, im * omegaC) }

    // Convert to digital poles via bilinear transform
    val digitalPoles = scaledPoles.map { case (re, im) => bilinear(re, im) }

    // All zeros at z = -1 (LP Butterworth)
    // Polynomial expansion from roots
    val a = rootsToPolynomial(digitalPoles)

    // Gain normalization: H(1) = 1 (DC gain = 1 for LP)
    val gain = a.zipWithIndex.map { case (c, i) =>
      c * math.pow(1.0, i)  // evaluate at z=1: sum of a coefficients
    }.sum

    val bGain = (0 to n).map { i =>
      // b coefficients for LP: (1 + z^{-1})^n
      binomialCoeff(n, i).toDouble / gain
    }.toArray

    (bGain, a)
  }

  private def rootsToPolynomial(roots: Array[(Double, Double)]): Array[Double] = {
    // Start with [1.0] and multiply by (1 - root_k * z^{-1}) for each root
    var poly = Array(1.0)
    for ((re, im) <- roots) {
      val newPoly = new Array[Double](poly.length + 1)
      for (i <- poly.indices) {
        newPoly(i)   += poly(i)
        newPoly(i+1) -= poly(i) * re  // real part of pole
      }
      poly = newPoly
    }
    poly
  }

  private def binomialCoeff(n: Int, k: Int): Long = {
    if (k == 0 || k == n) 1L
    else {
      var c = 1L
      val kk = math.min(k, n - k)
      for (i <- 0 until kk) c = c * (n - i) / (i + 1)
      c
    }
  }

  // Design the actual filter based on type
  private val (bCoeffs, aCoeffs): (Array[Double], Array[Double]) = {
    filterType match {
      case LowPass =>
        val fc = cutoff.swap.getOrElse(0.0)
        val omegaC = prewarp(fc)
        designLP(omegaC)

      case HighPass =>
        // HP = spectral inversion of LP
        val fc = cutoff.swap.getOrElse(0.0)
        val omegaC = prewarp(1.0 - fc)
        val (bLP, aLP) = designLP(omegaC)
        // Negate odd-index coefficients for HP transformation
        val bHP = bLP.zipWithIndex.map { case (b, i) =>
          if (i % 2 == 0) b else -b
        }
        val aHP = aLP.zipWithIndex.map { case (a, i) =>
          if (i % 2 == 0) a else -a
        }
        (bHP, aHP)

      case BandPass =>
        val (fLow, fHigh) = cutoff.getOrElse((0.0, 0.0))
        val oLow  = prewarp(fLow)
        val oHigh = prewarp(fHigh)
        // BP: cascade LP at oHigh with HP at oLow
        val (bLP, aLP) = designLP(oHigh)
        val (bHP, aHP) = {
          val (b0, a0) = designLP(1.0 - oLow / 2.0)
          val bH = b0.zipWithIndex.map { case (b, i) => if (i % 2 == 0) b else -b }
          val aH = a0.zipWithIndex.map { case (a, i) => if (i % 2 == 0) a else -a }
          (bH, aH)
        }
        // Convolve the two filter responses
        (convolve(bLP, bHP), convolve(aLP, aHP))
    }
  }

  private def convolve(a: Array[Double], b: Array[Double]): Array[Double] = {
    val result = new Array[Double](a.length + b.length - 1)
    for (i <- a.indices; j <- b.indices)
      result(i + j) += a(i) * b(j)
    result
  }

  /**
   * Applies the zero-phase forward-backward filter to a series.
   *
   * The effective order of the resulting filter is 2 * order.
   *
   * @param y input series
   * @return ButterworthResult with filtered series
   */
  def filter(y: Array[Double]): ButterworthResult = {
    val n = y.length
    require(n > bCoeffs.length, "series too short for filter order")

    // Forward pass
    val forward  = applyFilter(y, bCoeffs, aCoeffs)
    // Reverse and filter again
    val backward = applyFilter(forward.reverse, bCoeffs, aCoeffs)
    // Reverse back
    val result   = backward.reverse

    ButterworthResult(
      filtered    = result,
      order       = order,
      effectiveOrder = 2 * order,
      filterType  = filterType,
      cutoff      = cutoff
    )
  }

  /**
   * Applies a causal IIR filter using the difference equation:
   *   y(n) = b(0)*x(n) + b(1)*x(n-1) + ... - a(1)*y(n-1) - ...
   */
  private def applyFilter(
    x: Array[Double],
    b: Array[Double],
    a: Array[Double]
  ): Array[Double] = {
    val n      = x.length
    val nb     = b.length
    val na     = a.length
    val result = new Array[Double](n)

    for (i <- 0 until n) {
      var y = 0.0
      for (j <- 0 until nb)
        if (i - j >= 0) y += b(j) * x(i - j)
      for (j <- 1 until na)
        if (i - j >= 0) y -= a(j) * result(i - j)
      result(i) = y
    }
    result
  }
}

object ButterworthFilter {
  sealed trait FilterType
  case object LowPass  extends FilterType
  case object HighPass extends FilterType
  case object BandPass extends FilterType

  def lowPass(order: Int, cutoff: Double): ButterworthFilter =
    new ButterworthFilter(order, LowPass, Left(cutoff))

  def highPass(order: Int, cutoff: Double): ButterworthFilter =
    new ButterworthFilter(order, HighPass, Left(cutoff))

  def bandPass(order: Int, low: Double, high: Double): ButterworthFilter =
    new ButterworthFilter(order, BandPass, Right((low, high)))
}

case class ButterworthResult(
  filtered:       Array[Double],
  order:          Int,
  effectiveOrder: Int,
  filterType:     ButterworthFilter.FilterType,
  cutoff:         Either[Double, (Double, Double)]
) {
  override def toString: String = {
    val cutoffStr = cutoff match {
      case Left(f)       => f"$f%.3f"
      case Right((l, h)) => f"[$l%.3f, $h%.3f]"
    }
    s"Butterworth Filter ($filterType, order=$order, " +
    s"effectiveOrder=$effectiveOrder, cutoff=$cutoffStr)\n" +
    s"  Output length: ${filtered.length}\n"
  }
}
