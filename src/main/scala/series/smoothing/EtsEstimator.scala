package series.smoothing

import breeze.linalg._
import breeze.optimize.{LBFGS, DiffFunction}

/**
 * ETS model estimation via maximum likelihood.
 *
 * State initialization uses diffuse priors via the DiffuseKalmanFilter,
 * consistent with the Durbin-Koopman approach used elsewhere in this library.
 *
 * For the initial state:
 *   Level:    initialized to y(0) (first observation)
 *   Trend:    initialized to y(1) - y(0) (first difference)
 *   Seasonal: initialized from first full period of data
 *
 * Optimization: LBFGS with parameter box constraints enforced via
 * sigmoid/logit reparameterization.
 */
class EtsEstimator(spec: EtsSpec) {

  /**
   * Fits ETS model to a time series.
   *
   * @param y time series (length >= 2 * period for seasonal models)
   * @return EtsFit with parameter estimates, log-likelihood, AIC
   */
  def fit(y: Array[Double]): EtsFit = {
    require(y.length >= spec.period * 2 + 1,
      s"series too short for ${spec.name}")

    val init = initializeState(y)

    // Objective: negative log-likelihood
    val negLogLik: DiffFunction[DenseVector[Double]] =
      new DiffFunction[DenseVector[Double]] {
        def calculate(theta: DenseVector[Double]): (Double, DenseVector[Double]) = {
          val params = unpack(theta)
          val ll     = logLikelihood(y, params, init)
          // Numerical gradient
          val grad   = DenseVector.zeros[Double](theta.length)
          val h      = 1e-5
          for (i <- 0 until theta.length) {
            val thetaP = theta.copy; thetaP(i) += h
            val thetaM = theta.copy; thetaM(i) -= h
            val llP    = logLikelihood(y, unpack(thetaP), init)
            val llM    = logLikelihood(y, unpack(thetaM), init)
            grad(i)    = -(llP - llM) / (2 * h)
          }
          (-ll, grad)
        }
      }

    val x0      = pack(EtsParams.defaults(spec))
    val lbfgs   = new LBFGS[DenseVector[Double]](maxIter = 200, m = 7)
    val optX    = lbfgs.minimize(negLogLik, x0)
    val params  = unpack(optX)
    val ll      = logLikelihood(y, params, init)
    val n       = y.length
    val nParams = spec.nParams
    val aic     = -2 * ll + 2 * nParams
    val aicc    = aic + 2.0 * nParams * (nParams + 1) / (n - nParams - 1)
    val bic     = -2 * ll + math.log(n) * nParams

    // Final state sequence for diagnostics
    val states = filterStates(y, params, init)

    EtsFit(
      spec      = spec,
      params    = params,
      initState = init,
      states    = states,
      logLik    = ll,
      aic       = aic,
      aicc      = aicc,
      bic       = bic,
      n         = n,
      y         = y
    )
  }

  /** Initialize state from data (Hyndman-style backcasting approximation) */
  def initializeState(y: Array[Double]): DenseVector[Double] = {
    val state = DenseVector.zeros[Double](spec.stateDim)
    val p     = spec.period

    // Level: first observation
    state(0) = y(0)

    // Trend
    if (spec.hasTrend) {
      state(1) = if (y.length > 1) (y(math.min(p, y.length-1)) - y(0)) / p else 0.0
    }

    // Seasonal: deviations from mean of first period
    if (spec.hasSeasonal) {
      val sIdx  = 1 + (if (spec.hasTrend) 1 else 0)
      val nFull = math.min(y.length / p, 2) * p
      val mean  = y.take(nFull).sum / nFull
      spec.seasonal match {
        case AdditiveSeasonal =>
          for (j <- 0 until p)
            state(sIdx + j) = (if (j < y.length) y(j) - mean else 0.0)
        case MultiplicativeSeasonal =>
          for (j <- 0 until p)
            state(sIdx + j) = (if (j < y.length && math.abs(mean) > 1e-10)
              y(j) / mean else 1.0)
        case NoSeasonal =>
      }
    }

    state
  }

  /** Compute log-likelihood via innovation form */
  def logLikelihood(
    y:      Array[Double],
    params: EtsParams,
    init:   DenseVector[Double]
  ): Double = {
    val ss  = new EtsStateSpace(spec, params)
    var x   = init.copy
    val n   = y.length
    var ll  = 0.0
    val sig = math.sqrt(math.max(params.sigma2, 1e-10))

    for (t <- 0 until n) {
      val mu   = ss.w(x)
      val err  = spec.error match {
        case AdditiveError       => y(t) - mu
        case MultiplicativeError => (y(t) - mu) / math.max(math.abs(mu), 1e-10)
      }
      val scale = spec.error match {
        case AdditiveError       => sig
        case MultiplicativeError => math.abs(mu) * sig
      }
      ll += -0.5 * math.log(2 * math.Pi) - math.log(math.max(scale, 1e-10)) -
            0.5 * (err * err) / math.max(params.sigma2, 1e-10)
      x = ss.transition(x, err / math.max(sig, 1e-10))
    }
    ll
  }

  /** Extract state sequence */
  private def filterStates(
    y:      Array[Double],
    params: EtsParams,
    init:   DenseVector[Double]
  ): Array[DenseVector[Double]] = {
    val ss     = new EtsStateSpace(spec, params)
    var x      = init.copy
    val states = new Array[DenseVector[Double]](y.length + 1)
    states(0)  = x.copy
    val sig    = math.sqrt(math.max(params.sigma2, 1e-10))

    for (t <- y.indices) {
      val mu  = ss.w(x)
      val err = spec.error match {
        case AdditiveError       => y(t) - mu
        case MultiplicativeError => (y(t) - mu) / math.max(math.abs(mu), 1e-10)
      }
      x = ss.transition(x, err / math.max(sig, 1e-10))
      states(t + 1) = x.copy
    }
    states
  }

  // ── Parameter packing / unpacking ─────────────────────────────────────────

  /**
   * Pack parameters into unconstrained space for LBFGS.
   * Uses logit transform: theta = log(p / (1-p)) for p in (a, b)
   */
  private def pack(p: EtsParams): DenseVector[Double] = {
    val v = scala.collection.mutable.ArrayBuffer[Double]()
    v += logit01(p.alpha)
    if (spec.hasTrend)    v += logit01(p.beta / p.alpha)  // beta in (0, alpha)
    if (spec.hasSeasonal) v += logit01(p.gamma)
    if (spec.isDamped)    v += logit(p.phi, 0.8, 0.98)
    v += math.log(p.sigma2)  // sigma2 > 0 via exp
    DenseVector(v.toArray)
  }

  private def unpack(theta: DenseVector[Double]): EtsParams = {
    var i = 0
    val alpha = sigmoid01(theta(i)); i += 1
    val beta  = if (spec.hasTrend) {
      val r = sigmoid01(theta(i)); i += 1
      r * alpha
    } else 0.0
    val gamma = if (spec.hasSeasonal) {
      val r = sigmoid01(theta(i)); i += 1
      math.min(r, 1.0 - alpha)
    } else 0.0
    val phi   = if (spec.isDamped) {
      val r = sigmoidAB(theta(i), 0.8, 0.98); i += 1
      r
    } else 1.0
    val sigma2 = math.exp(theta(i))
    EtsParams(alpha, beta, gamma, phi, sigma2)
  }

  private def logit01(p: Double): Double = {
    val pp = math.max(0.001, math.min(0.999, p))
    math.log(pp / (1 - pp))
  }

  private def logit(p: Double, lo: Double, hi: Double): Double = {
    val pp = math.max(lo + 0.001, math.min(hi - 0.001, p))
    math.log((pp - lo) / (hi - pp))
  }

  private def sigmoid01(x: Double): Double = 1.0 / (1.0 + math.exp(-x))

  private def sigmoidAB(x: Double, lo: Double, hi: Double): Double = {
    val s = sigmoid01(x)
    lo + s * (hi - lo)
  }
}
