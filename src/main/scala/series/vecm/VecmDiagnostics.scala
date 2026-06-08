package series.vecm

import breeze.linalg._
import breeze.stats.distributions.{ChiSquared, RandBasis}
import series.stats.AutoCorrelation

class VecmDiagnostics(fit: VecmFit) {

  implicit val randBasis: RandBasis = RandBasis.systemSeed

  private val resid = fit.residuals
  private val n     = resid.rows
  private val k     = fit.k

  def ljungBox(h: Int): LjungBoxMVResult = {
    val c0 = (resid.t * resid) / n.toDouble

    val Q = (1 to h).map { lag =>
      val cl = DenseMatrix.tabulate(k, k) { (i, j) =>
        (0 until n - lag).map(t =>
          resid(t, i) * resid(t + lag, j)
        ).sum / n.toDouble
      }
      val d   = diag(diag(c0).map(math.sqrt))
      val dInv = inv(d)
      val rl  = dInv * cl * dInv
      val traceRl = trace(rl.t * rl)
      n * (n + 2) * traceRl / (n - lag).toDouble
    }.sum

    val df     = h * k * k
    val pValue = 1.0 - new ChiSquared(df).cdf(Q)
    LjungBoxMVResult(Q, h, df, pValue)
  }

  def jarqueBera(): MultiNormalityResult = {
    val sigma  = fit.Sigma.copy
    val sigInv = inv(sigma)
    val zMat   = resid * sigInv

    val mu = DenseVector.tabulate(k)(j =>
      zMat(::, j).toArray.sum / n
    )

    var b1 = 0.0
    var b2 = 0.0

    for (t <- 0 until n) {
      val zt  = zMat(t, ::).t.copy - mu
      val mah = zt.t * sigInv * zt
      b2 += mah * mah
      for (s <- 0 until n) {
        val zs = zMat(s, ::).t.copy - mu
        val g  = zt.t * sigInv * zs
        b1 += g * g * g
      }
    }

    b1 /= (n * n).toDouble
    b2 /= n.toDouble

    val jb1    = n * b1 / 6.0
    val jb2    = n * (b2 - k * (k + 2)) * (b2 - k * (k + 2)) / (8.0 * k * (k + 2))
    val jb     = jb1 + jb2
    val df     = k * (k + 1) * (k + 2) / 6 + k
    val pValue = 1.0 - new ChiSquared(df).cdf(jb)

    MultiNormalityResult(jb, b1, b2, df, pValue)
  }

  def archLM(lags: Int): Array[ArchLMResult] = {
    (0 until k).map { j =>
      val eArr: Array[Double] = Array.tabulate(n)(t => fit.residuals(t, j))
      val e2: Array[Double]   = eArr.map(x => x * x)
      val nR    = n - lags
      val y     = e2.drop(lags)
      val mu    = y.sum / nR

      val xData = Array.tabulate(nR, lags + 1) { (t, l) =>
        if (l == 0) 1.0 else e2(t + lags - l)
      }
      val xm    = DenseMatrix(xData.toIndexedSeq: _*)
      val yv    = DenseVector(y)
      val xtx   = xm.t * xm
      val beta  = inv(xtx) * (xm.t * yv)
      val resid = yv - xm * beta

      val ssTot  = y.map(yi => math.pow(yi - mu, 2)).sum
      val ssRes  = resid.toArray.map(r => r * r).sum
      val r2     = 1.0 - ssRes / math.max(ssTot, 1e-10)
      val lmStat = nR * r2
      val pValue = 1.0 - new ChiSquared(lags).cdf(lmStat)
      ArchLMResult(j, lmStat, lags, pValue)
    }.toArray
  }

  def persistenceConvergence(h: Int): Boolean = {
    val irf      = new VecmIRF(fit)
    val profiles = irf.persistenceProfiles(h)
    profiles.forall { profile =>
      profile.last < profile.head * 0.5
    }
  }

  def report(h: Int = 10, archLags: Int = 5): String = {
    val lb   = ljungBox(h)
    val jb   = jarqueBera()
    val arch = archLM(archLags)
    val pc   = persistenceConvergence(h * 5)

    val sb = new StringBuilder
    sb.append("VECM Diagnostic Report\n======================\n\n")
    sb.append(s"Multivariate Ljung-Box (lag=$h):\n")
    sb.append(f"  Q = ${lb.statistic%.4f}, df = ${lb.df}, p-value = ${lb.pValue%.4f}\n")
    sb.append(s"  ${if (lb.pValue > 0.05) "No evidence of autocorrelation" else "Evidence of autocorrelation"}\n\n")
    sb.append("Multivariate Jarque-Bera:\n")
    sb.append(f"  JB = ${jb.statistic%.4f}, df = ${jb.df}, p-value = ${jb.pValue%.4f}\n")
    sb.append(s"  ${if (jb.pValue > 0.05) "No evidence against normality" else "Evidence against normality"}\n\n")
    sb.append(s"ARCH-LM Tests (lags=$archLags):\n")
    arch.foreach { a =>
      sb.append(f"  y${a.equation+1}: LM = ${a.statistic%.4f}, p-value = ${a.pValue%.4f}\n")
    }
    sb.append(s"\nPersistence Profiles (h=${h*5}):\n  Converging: $pc\n")
    sb.toString
  }
}

case class LjungBoxMVResult(statistic: Double, lag: Int, df: Int, pValue: Double) {
  def isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha
}

case class MultiNormalityResult(statistic: Double, skewness: Double, kurtosis: Double, df: Int, pValue: Double) {
  def isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha
}

case class ArchLMResult(equation: Int, statistic: Double, lags: Int, pValue: Double) {
  def isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha
}
