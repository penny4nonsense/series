package series.garch

/**
 * Results from GARCH(p,q) estimation.
 *
 * @param omega    intercept
 * @param alpha    ARCH coefficients
 * @param beta     GARCH coefficients
 * @param omegaSE  standard error for omega
 * @param alphaSEs standard errors for alpha
 * @param betaSEs  standard errors for beta
 * @param logLik   log likelihood at estimates
 * @param n        number of observations
 * @param sigma2   fitted conditional variances
 * @param residuals input residual series
 * @param p        GARCH order
 * @param q        ARCH order
 */
case class GarchFit(
  omega:     Double,
  alpha:     Array[Double],
  beta:      Array[Double],
  omegaSE:   Double,
  alphaSEs:  Array[Double],
  betaSEs:   Array[Double],
  logLik:    Double,
  n:         Int,
  sigma2:    Array[Double],
  residuals: Array[Double],
  p:         Int,
  q:         Int
) {
  private val k = 1 + q + p

  def aic:  Double = -2 * logLik + 2 * k
  def aicc: Double = aic + (2.0 * k * (k + 1)) / (n - k - 1)
  def bic:  Double = -2 * logLik + math.log(n) * k

  /** Persistence — sum of alpha and beta. Should be < 1 for stationarity. */
  def persistence: Double = alpha.sum + beta.sum

  /** Unconditional variance implied by model parameters. */
  def unconditionalVariance: Double =
    omega / math.max(1.0 - persistence, 1e-8)

  /** Standardized residuals eps_t / sigma_t. */
  def standardizedResiduals: Array[Double] =
    residuals.zip(sigma2).map { case (e, s) => e / math.sqrt(s) }

  /**
   * Produces h-step ahead conditional variance and volatility forecasts.
   * Convenience wrapper over GarchForecaster — the common entry point.
   */
  def forecast(h: Int): GarchForecast = new GarchForecaster(this).forecast(h)

  override def toString: String = {
    val header = f"  ${""}%-10s  ${"estimate"}%10s  ${"std.err"}%10s"

    val omegaRow = f"  ${"omega"}%-10s  $omega%10.6f  $omegaSE%10.6f"
    val alphaRows = alpha.zip(alphaSEs).zipWithIndex.map { case ((a, se), i) =>
      f"  ${s"alpha${i+1}"}%-10s  $a%10.6f  $se%10.6f"
    }
    val betaRows = beta.zip(betaSEs).zipWithIndex.map { case ((b, se), i) =>
      f"  ${s"beta${i+1}"}%-10s  $b%10.6f  $se%10.6f"
    }

    val rows = (omegaRow +: alphaRows ++: betaRows).mkString("\n")

    s"""GARCH($p,$q) Fit:
       |$header
       |$rows
       |  persistence        = ${f"$persistence%.6f"}
       |  unconditional var  = ${f"$unconditionalVariance%.6f"}
       |  logLik             = ${f"$logLik%.4f"}
       |  AIC                = ${f"$aic%.4f"}
       |  AICc               = ${f"$aicc%.4f"}
       |  BIC                = ${f"$bic%.4f"}
       |""".stripMargin
  }
}
