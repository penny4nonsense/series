package series.smoothing

import breeze.linalg._

/**
 * Fitted ETS model results.
 */
case class EtsFit(
  spec:      EtsSpec,
  params:    EtsParams,
  initState: DenseVector[Double],
  states:    Array[DenseVector[Double]],
  logLik:    Double,
  aic:       Double,
  aicc:      Double,
  bic:       Double,
  n:         Int,
  y:         Array[Double]
) {
  /** One-step-ahead fitted values */
  lazy val fitted: Array[Double] = {
    val ss = new EtsStateSpace(spec, params)
    states.take(n).map(ss.w)
  }

  /** Residuals (innovation errors) */
  lazy val residuals: Array[Double] = Array.tabulate(n) { t =>
    val mu = fitted(t)
    spec.error match {
      case AdditiveError       => y(t) - mu
      case MultiplicativeError => (y(t) - mu) / math.max(math.abs(mu), 1e-10)
    }
  }

  override def toString: String = {
    val sb = new StringBuilder
    sb.append(s"${spec.name} Model (n=$n)\n")
    sb.append(f"  alpha  = ${params.alpha}%.4f\n")
    if (spec.hasTrend)    sb.append(f"  beta   = ${params.beta}%.4f\n")
    if (spec.isDamped)    sb.append(f"  phi    = ${params.phi}%.4f\n")
    if (spec.hasSeasonal) sb.append(f"  gamma  = ${params.gamma}%.4f\n")
    sb.append(f"  sigma2 = ${params.sigma2}%.4f\n")
    sb.append(f"  logLik = $logLik%.4f\n")
    sb.append(f"  AIC    = $aic%.4f\n")
    sb.append(f"  AICc   = $aicc%.4f\n")
    sb.append(f"  BIC    = $bic%.4f\n")
    sb.toString
  }
}
