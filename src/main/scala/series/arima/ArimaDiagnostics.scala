package series.arima

import breeze.linalg._
import breeze.stats.distributions.{ChiSquared, RandBasis}
import series.kalman._

/**
 * Diagnostic statistics for fitted ARIMA and SARIMA models.
 *
 * Residuals are extracted from the innovations of the Kalman filter —
 * the one-step-ahead prediction errors. Under correct specification
 * these should be approximately iid N(0, sigma2).
 *
 * @param km      KalmanModel built from fitted parameters
 * @param initial DiffuseKalmanState for initialization
 * @param y       observed series (differenced)
 */
class ArimaDiagnostics(
                        km:      KalmanModel,
                        initial: DiffuseKalmanState,
                        y:       Array[Double]
                      ) {

  implicit val randBasis: RandBasis = RandBasis.systemSeed

  private val (innovations, innovationVariances): (Array[Double], Array[Double]) = {
    val obs    = y.map(v => DenseVector(v))
    var state  = initial
    val innov  = Array.fill(y.length)(0.0)
    val innovV = Array.fill(y.length)(0.0)

    for (i <- y.indices) {
      val predicted = DiffuseKalmanFilter.predict(state, km)
      val yhat      = km.H * predicted.x
      val Fmat      = km.H * predicted.Pstar * km.H.t
      innov(i)  = y(i) - yhat(0)
      innovV(i) = Fmat(0, 0) + km.R(0, 0)
      state = DiffuseKalmanFilter.step(state, km, DenseVector(y(i)))
    }
    (innov, innovV)
  }

  val standardizedResiduals: Array[Double] =
    innovations.zip(innovationVariances).map { case (e, v) =>
      if (v > 0.0) e / math.sqrt(v) else Double.NaN
    }

  def residualACF(maxLag: Int): Array[Double] = {
    val r  = standardizedResiduals.filter(!_.isNaN)
    val n  = r.length
    val mu = r.sum / n
    val v  = r.map(x => x - mu)
    val c0 = v.map(x => x * x).sum / n

    (1 to maxLag).map { lag =>
      val c = (0 until n - lag).map(i => v(i) * v(i + lag)).sum / n
      c / c0
    }.toArray
  }

  def ljungBox(h: Int): LjungBoxResult = {
    val r   = standardizedResiduals.filter(!_.isNaN)
    val n   = r.length
    val acf = residualACF(h)

    val Q = n * (n + 2) * acf.zipWithIndex.map { case (rk, k) =>
      (rk * rk) / (n - (k + 1))
    }.sum

    val pValue = 1.0 - new ChiSquared(h).cdf(Q)
    LjungBoxResult(Q, h, pValue)
  }

  def jarqueBera(): JarqueBeraResult = {
    val r  = standardizedResiduals.filter(!_.isNaN)
    val n  = r.length
    val mu = r.sum / n

    val m2 = r.map(x => math.pow(x - mu, 2)).sum / n
    val m3 = r.map(x => math.pow(x - mu, 3)).sum / n
    val m4 = r.map(x => math.pow(x - mu, 4)).sum / n

    val skewness = m3 / math.pow(m2, 1.5)
    val kurtosis = m4 / (m2 * m2) - 3.0

    val JB     = n / 6.0 * (skewness * skewness + kurtosis * kurtosis / 4.0)
    val pValue = 1.0 - new ChiSquared(2).cdf(JB)

    JarqueBeraResult(JB, skewness, kurtosis, pValue)
  }

  def residualSummary(): ResidualSummary = {
    val r      = standardizedResiduals.filter(!_.isNaN)
    val n      = r.length
    val mu     = r.sum / n
    val v      = r.map(x => math.pow(x - mu, 2)).sum / (n - 1)
    val sorted = r.sorted

    ResidualSummary(
      mean     = mu,
      variance = v,
      min      = sorted.head,
      max      = sorted.last,
      q25      = sorted((n * 0.25).toInt),
      median   = sorted(n / 2),
      q75      = sorted((n * 0.75).toInt),
      n        = n
    )
  }

  def report(ljungBoxLag: Int = 10): String = {
    val lb = ljungBox(ljungBoxLag)
    val jb = jarqueBera()
    val rs = residualSummary()

    s"""Diagnostic Report
       |=================
       |
       |Residual Summary (standardized):
       |  n        = ${rs.n}
       |  mean     = ${f"${rs.mean}%.4f"}
       |  variance = ${f"${rs.variance}%.4f"}
       |  min      = ${f"${rs.min}%.4f"}
       |  Q25      = ${f"${rs.q25}%.4f"}
       |  median   = ${f"${rs.median}%.4f"}
       |  Q75      = ${f"${rs.q75}%.4f"}
       |  max      = ${f"${rs.max}%.4f"}
       |
       |Ljung-Box Test (lag = ${lb.lag}):
       |  Q        = ${f"${lb.statistic}%.4f"}
       |  df       = ${lb.lag}
       |  p-value  = ${f"${lb.pValue}%.4f"}
       |  ${if (lb.pValue > 0.05) "No evidence of autocorrelation" else "Evidence of autocorrelation"}
       |
       |Jarque-Bera Test:
       |  JB       = ${f"${jb.statistic}%.4f"}
       |  skewness = ${f"${jb.skewness}%.4f"}
       |  kurtosis = ${f"${jb.excessKurtosis}%.4f"}
       |  p-value  = ${f"${jb.pValue}%.4f"}
       |  ${if (jb.pValue > 0.05) "No evidence against normality" else "Evidence against normality"}
       |""".stripMargin
  }
}

case class LjungBoxResult(
                           statistic: Double,
                           lag:       Int,
                           pValue:    Double
                         ) {
  def isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha
  override def toString: String =
    f"Ljung-Box Q($lag) = $statistic%.4f, p-value = $pValue%.4f"
}

case class JarqueBeraResult(
                             statistic:      Double,
                             skewness:       Double,
                             excessKurtosis: Double,
                             pValue:         Double
                           ) {
  def isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha
  override def toString: String =
    f"Jarque-Bera = $statistic%.4f, skewness = $skewness%.4f, " +
      f"excess kurtosis = $excessKurtosis%.4f, p-value = $pValue%.4f"
}

case class ResidualSummary(
                            mean:     Double,
                            variance: Double,
                            min:      Double,
                            max:      Double,
                            q25:      Double,
                            median:   Double,
                            q75:      Double,
                            n:        Int
                          )