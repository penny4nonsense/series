package series.garch

import breeze.linalg._
import breeze.optimize._

/**
 * GARCH(p,q) model for conditional variance.
 *
 * The conditional variance equation:
 *   sigma2_t = omega + sum_i alpha_i * eps_{t-i}^2
 *                    + sum_j beta_j  * sigma2_{t-j}
 *
 * ARCH(q) is the special case p=0.
 *
 * Estimation via exact MLE using Breeze LBFGS.
 * Parameters are transformed to enforce positivity and
 * stationarity constraints during optimization.
 *
 * @param p order of GARCH terms (lagged variances)
 * @param q order of ARCH terms (lagged squared residuals)
 */
case class GarchModel(p: Int, q: Int) {
  require(p >= 0, "p must be non-negative")
  require(q >= 0, "q must be non-negative")
  require(p + q > 0, "At least one of p, q must be positive")

  val nParams: Int = 1 + q + p  // omega, alpha_1..alpha_q, beta_1..beta_p

  /**
   * Computes conditional variance sequence given parameters and residuals.
   *
   * Initializes sigma2_0 with unconditional variance.
   *
   * @param eps    residual series
   * @param omega  intercept
   * @param alpha  ARCH coefficients (length q)
   * @param beta   GARCH coefficients (length p)
   * @return conditional variance sequence (same length as eps)
   */
  def conditionalVariances(
                            eps:   Array[Double],
                            omega: Double,
                            alpha: Array[Double],
                            beta:  Array[Double]
                          ): Array[Double] = {
    val n = eps.length

    // Unconditional variance as initial value
    val unconditional = omega / math.max(1.0 - alpha.sum - beta.sum, 1e-6)
    val sigma2 = Array.fill(n)(unconditional)

    for (t <- 1 until n) {
      var v = omega
      // ARCH terms
      for (i <- 0 until math.min(q, t))
        v += alpha(i) * eps(t - 1 - i) * eps(t - 1 - i)
      // GARCH terms
      for (j <- 0 until math.min(p, t))
        v += beta(j) * sigma2(t - 1 - j)
      sigma2(t) = math.max(v, 1e-8)  // floor to avoid numerical issues
    }

    sigma2
  }

  /**
   * Gaussian log likelihood for GARCH model.
   *
   * @param eps    residual series
   * @param omega  intercept
   * @param alpha  ARCH coefficients
   * @param beta   GARCH coefficients
   * @return log likelihood
   */
  def logLikelihood(
                     eps:   Array[Double],
                     omega: Double,
                     alpha: Array[Double],
                     beta:  Array[Double]
                   ): Double = {
    val sigma2 = conditionalVariances(eps, omega, alpha, beta)
    val n      = eps.length

    -0.5 * (n * math.log(2 * math.Pi) +
      (0 until n).map(t =>
        math.log(sigma2(t)) + eps(t) * eps(t) / sigma2(t)
      ).sum)
  }

  /**
   * Unpacks parameter vector into omega, alpha, beta.
   *
   * Transforms:
   *   omega = exp(params(0))                        — positivity
   *   alpha = softmax-scaled to sum < 1 with beta   — stationarity
   *   beta  = softmax-scaled to sum < 1 with alpha  — stationarity
   */
  def unpack(params: DenseVector[Double]): (Double, Array[Double], Double, Array[Double]) = {
    val omega = math.exp(params(0))

    // Use logistic transforms for alpha and beta
    // then scale to ensure sum(alpha) + sum(beta) < 1
    val rawAlpha = params(1 until 1 + q).toArray.map(x => math.exp(x))
    val rawBeta  = params(1 + q until 1 + q + p).toArray.map(x => math.exp(x))

    val totalRaw = rawAlpha.sum + rawBeta.sum + 1.0  // +1 ensures sum < 1
    val scale    = 0.999 / totalRaw

    val alpha = rawAlpha.map(_ * scale)
    val beta  = rawBeta.map(_ * scale)

    (omega, alpha, rawAlpha.sum * scale, beta)
  }

  /**
   * Packs omega, alpha, beta into unconstrained parameter vector.
   */
  def pack(
            omega: Double,
            alpha: Array[Double],
            beta:  Array[Double]
          ): DenseVector[Double] = {
    val total = alpha.sum + beta.sum
    val scale = if (total > 0.0) 0.999 / total else 1.0
    DenseVector(
      Array(math.log(omega)) ++
        alpha.map(a => math.log(math.max(a * scale, 1e-8))) ++
        beta.map(b => math.log(math.max(b * scale, 1e-8)))
    )
  }

  /**
   * Negative log likelihood for optimization.
   */
  private def negLogLik(
                         params: DenseVector[Double],
                         eps:    Array[Double]
                       ): Double = {
    try {
      val (omega, alpha, _, beta) = unpack(params)
      -logLikelihood(eps, omega, alpha, beta)
    } catch {
      case _: Exception => Double.MaxValue
    }
  }

  /**
   * Numerical gradient of negative log likelihood.
   */
  private def numericalGradient(
                                 params: DenseVector[Double],
                                 eps:    Array[Double]
                               ): DenseVector[Double] = {
    val g = DenseVector.zeros[Double](params.length)
    val h = 1e-5
    for (i <- 0 until params.length) {
      val pp = params.copy; pp(i) += h
      val pm = params.copy; pm(i) -= h
      g(i) = (negLogLik(pp, eps) - negLogLik(pm, eps)) / (2 * h)
    }
    g
  }

  /**
   * Numerical Hessian for standard errors.
   */
  private def numericalHessian(
                                params: DenseVector[Double],
                                eps:    Array[Double]
                              ): DenseMatrix[Double] = {
    val n = params.length
    val H = DenseMatrix.zeros[Double](n, n)
    val h = 1e-4
    for (i <- 0 until n; j <- 0 until n) {
      val pp = params.copy; pp(i) += h; pp(j) += h
      val pm = params.copy; pm(i) += h; pm(j) -= h
      val mp = params.copy; mp(i) -= h; mp(j) += h
      val mm = params.copy; mm(i) -= h; mm(j) -= h
      H(i, j) = (negLogLik(pp, eps) - negLogLik(pm, eps) -
        negLogLik(mp, eps) + negLogLik(mm, eps)) / (4 * h * h)
    }
    H
  }

  /**
   * Fits GARCH(p,q) by maximum likelihood using Breeze LBFGS.
   *
   * @param eps residual series (mean-corrected returns or ARIMA residuals)
   * @return GarchFit with parameter estimates and diagnostics
   */
  def fit(eps: Array[Double]): GarchFit = {
    val n = eps.length

    // Starting values — small ARCH/GARCH effects, variance as omega
    val varEps = eps.map(e => e * e).sum / n
    val initAlpha = Array.fill(q)(0.05)
    val initBeta  = Array.fill(p)(0.05)
    val initOmega = varEps * (1.0 - initAlpha.sum - initBeta.sum)

    val initParams = pack(initOmega, initAlpha, initBeta)

    // LBFGS optimization
    val objective = new DiffFunction[DenseVector[Double]] {
      def calculate(params: DenseVector[Double]) = {
        val f = negLogLik(params, eps)
        val g = numericalGradient(params, eps)
        (f, g)
      }
    }

    val optParams = new LBFGS[DenseVector[Double]](maxIter=500, m=10)
      .minimize(objective, initParams)

    val (omega, alpha, _, beta) = unpack(optParams)
    val ll      = logLikelihood(eps, omega, alpha, beta)
    val sigma2  = conditionalVariances(eps, omega, alpha, beta)

    // Standard errors
    val ses = try {
      val H    = numericalHessian(optParams, eps)
      val Hinv = inv(H)
      (0 until optParams.length).map(i =>
        math.sqrt(math.abs(Hinv(i, i)))
      ).toArray
    } catch {
      case _: Exception => Array.fill(nParams)(Double.NaN)
    }

    // Delta method SEs: omega SE from log transform
    val omegaSE = omega * ses(0)
    val alphaSEs = alpha.zipWithIndex.map { case (a, i) => a * ses(1 + i) }
    val betaSEs  = beta.zipWithIndex.map  { case (b, j) => b * ses(1 + q + j) }

    GarchFit(
      omega      = omega,
      alpha      = alpha,
      beta       = beta,
      omegaSE    = omegaSE,
      alphaSEs   = alphaSEs,
      betaSEs    = betaSEs,
      logLik     = ll,
      n          = n,
      sigma2     = sigma2,
      residuals  = eps,
      p          = p,
      q          = q
    )
  }
}