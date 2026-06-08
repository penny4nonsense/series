package series.garch

import breeze.stats.distributions.ChiSquared
import series.stats.AutoCorrelation

/**
 * Diagnostic tests for fitted GARCH models.
 *
 * Tests are applied to standardized residuals z_t = eps_t / sigma_t.
 * Under correct specification these should be approximately iid N(0,1).
 *
 * Key tests:
 *   Ljung-Box on z_t       — tests for remaining mean autocorrelation
 *   Ljung-Box on z_t^2     — tests for remaining variance autocorrelation
 *   ARCH-LM test           — Engle's LM test for remaining ARCH effects
 *   Jarque-Bera            — normality of standardized residuals
 */
class GarchDiagnostics(fit: GarchFit) {

  implicit val randBasis: breeze.stats.distributions.RandBasis =
    breeze.stats.distributions.RandBasis.systemSeed

  private val z  = fit.standardizedResiduals
  private val z2 = z.map(x => x * x)
  private val n  = z.length

  /**
   * Ljung-Box test on standardized residuals.
   * Tests for remaining autocorrelation in the mean.
   *
   * @param h maximum lag
   */
  def ljungBoxResiduals(h: Int): LMResult = {
    val acf = AutoCorrelation.acf(z, h)
    val Q   = n * (n + 2) * acf.zipWithIndex.map { case (rk, k) =>
      (rk * rk) / (n - (k + 1))
    }.sum
    val pValue = 1.0 - new ChiSquared(h).cdf(Q)
    LMResult(Q, h, pValue, "Ljung-Box on standardized residuals")
  }

  /**
   * Ljung-Box test on squared standardized residuals.
   * Tests for remaining ARCH effects.
   *
   * @param h maximum lag
   */
  def ljungBoxSquared(h: Int): LMResult = {
    val acf = AutoCorrelation.acf(z2, h)
    val Q   = n * (n + 2) * acf.zipWithIndex.map { case (rk, k) =>
      (rk * rk) / (n - (k + 1))
    }.sum
    val pValue = 1.0 - new ChiSquared(h).cdf(Q)
    LMResult(Q, h, pValue, "Ljung-Box on squared standardized residuals")
  }

  /**
   * Engle's ARCH-LM test for remaining ARCH effects.
   *
   * Regresses z_t^2 on its own lags and tests joint significance.
   * Under H0 (no remaining ARCH) the LM statistic is chi-squared(lags).
   *
   * @param lags number of lags
   */
  def archLM(lags: Int): LMResult = {
    require(lags > 0 && lags < n, "lags must be positive and less than n")

    val nReg = n - lags
    val y    = z2.drop(lags)
    val mu   = y.sum / nReg

    // Build X matrix: constant + lagged z^2
    val X = Array.tabulate(nReg, lags + 1) { (i, j) =>
      if (j == 0) 1.0 else z2(i + lags - j)
    }

    // OLS
    import breeze.linalg._
    val Xm = DenseMatrix(X.toIndexedSeq: _*)
    val yv    = DenseVector(y)
    val beta  = inv(Xm.t * Xm) * Xm.t * yv
    val resid = yv - Xm * beta

    // R-squared
    val ssTot = y.map(yi => math.pow(yi - mu, 2)).sum
    val ssRes = (resid.toArray).map(r => r * r).sum
    val r2    = 1.0 - ssRes / math.max(ssTot, 1e-10)

    val lmStat = nReg * r2
    val pValue = 1.0 - new ChiSquared(lags).cdf(lmStat)

    LMResult(lmStat, lags, pValue, "ARCH-LM test")
  }

  /**
   * Jarque-Bera normality test on standardized residuals.
   */
  def jarqueBera(): JBResult = {
    val mu = z.sum / n
    val m2 = z.map(x => math.pow(x - mu, 2)).sum / n
    val m3 = z.map(x => math.pow(x - mu, 3)).sum / n
    val m4 = z.map(x => math.pow(x - mu, 4)).sum / n

    val skewness = m3 / math.pow(m2, 1.5)
    val kurtosis = m4 / (m2 * m2) - 3.0

    val JB     = n / 6.0 * (skewness * skewness + kurtosis * kurtosis / 4.0)
    val pValue = 1.0 - new ChiSquared(2).cdf(JB)

    JBResult(JB, skewness, kurtosis, pValue)
  }

  /**
   * Summary statistics for standardized residuals.
   */
  def residualSummary(): ResidualStats = {
    val mu     = z.sum / n
    val v      = z.map(x => math.pow(x - mu, 2)).sum / (n - 1)
    val sorted = z.sorted
    ResidualStats(
      mean     = mu,
      variance = v,
      min      = sorted.head,
      max      = sorted.last,
      q25      = sorted((n * 0.25).toInt),
      median   = sorted(n / 2),
      q75      = sorted((n * 0.75).toInt)
    )
  }

  /**
   * Full diagnostic report.
   */
  def report(lags: Int = 10): String = {
    val lb1 = ljungBoxResiduals(lags)
    val lb2 = ljungBoxSquared(lags)
    val lm  = archLM(lags)
    val jb  = jarqueBera()
    val rs  = residualSummary()

    s"""GARCH Diagnostic Report
       |=======================
       |
       |Standardized Residual Summary:
       |  n        = $n
       |  mean     = ${f"${rs.mean}%.4f"}
       |  variance = ${f"${rs.variance}%.4f"}
       |  min      = ${f"${rs.min}%.4f"}
       |  Q25      = ${f"${rs.q25}%.4f"}
       |  median   = ${f"${rs.median}%.4f"}
       |  Q75      = ${f"${rs.q75}%.4f"}
       |  max      = ${f"${rs.max}%.4f"}
       |
       |${lb1.toString}
       |${lb2.toString}
       |${lm.toString}
       |
       |Jarque-Bera Test:
       |  JB            = ${f"${jb.statistic}%.4f"}
       |  skewness      = ${f"${jb.skewness}%.4f"}
       |  excess kurt.  = ${f"${jb.excessKurtosis}%.4f"}
       |  p-value       = ${f"${jb.pValue}%.4f"}
       |  ${if (jb.pValue > 0.05) "No evidence against normality" else "Evidence against normality"}
       |""".stripMargin
  }
}

// Result types

case class LMResult(
                     statistic:   Double,
                     lag:         Int,
                     pValue:      Double,
                     description: String
                   ) {
  def isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha
  override def toString: String =
    s"""$description:
       |  statistic = ${f"$statistic%.4f"}
       |  df        = $lag
       |  p-value   = ${f"$pValue%.4f"}
       |  ${if (isSignificant()) "Significant at 5%" else "Not significant at 5%"}"""
      .stripMargin
}

case class JBResult(
                     statistic:      Double,
                     skewness:       Double,
                     excessKurtosis: Double,
                     pValue:         Double
                   ) {
  def isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha
}

case class ResidualStats(
                          mean:     Double,
                          variance: Double,
                          min:      Double,
                          max:      Double,
                          q25:      Double,
                          median:   Double,
                          q75:      Double
                        )