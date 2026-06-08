package series.stats

import breeze.linalg._
import breeze.linalg.eigSym

object CointegrationTests {

  def engleGranger(
                    y:       Array[Double],
                    x:       Array[Double],
                    maxLag:  Int = -1
                  ): EngleGrangerResult = {
    require(y.length == x.length, "Series must have equal length")

    val n     = y.length
    val xMat  = DenseMatrix.tabulate(n, 2)((i, j) => if (j == 0) 1.0 else x(i))
    val yv    = DenseVector(y)
    val beta  = inv(xMat.t * xMat) * xMat.t * yv
    val resid = (yv - xMat * beta).toArray

    val lags   = if (maxLag < 0) UnitRootTests.adf(resid, 0, "none").lagOrder else maxLag
    val adfRes = UnitRootTests.adf(resid, lags, "none")

    EngleGrangerResult(
      statistic       = adfRes.statistic,
      lagOrder        = lags,
      alpha           = beta(0),
      beta            = beta(1),
      residuals       = resid,
      criticalValue1  = -3.896,
      criticalValue5  = -3.338,
      criticalValue10 = -3.046
    )
  }

  def johansen(
                Y:      DenseMatrix[Double],
                lags:   Int    = 1,
                regime: String = "drift"
              ): JohansenResult = {
    require(lags > 0, "lags must be positive")
    require(Seq("none", "drift", "trend").contains(regime),
      "regime must be 'none', 'drift', or 'trend'")

    val n    = Y.rows
    val k    = Y.cols
    val dY   = DenseMatrix.tabulate(n - 1, k)((i, j) => Y(i + 1, j) - Y(i, j))
    val nObs = n - 1 - lags

    val r0mat = DenseMatrix.zeros[Double](nObs, k)
    val r1mat = DenseMatrix.zeros[Double](nObs, k)

    for (i <- 0 until nObs) {
      val t = i + lags
      for (j <- 0 until k) {
        r0mat(i, j) = dY(t, j)
        r1mat(i, j) = Y(t, j)
      }
    }

    val nDet = regime match {
      case "none"  => 0
      case "drift" => 1
      case "trend" => 2
    }
    val nAux = k * lags + nDet
    val zMat = DenseMatrix.zeros[Double](nObs, nAux)

    for (i <- 0 until nObs) {
      val t = i + lags
      var col = 0
      if (regime == "drift" || regime == "trend") { zMat(i, col) = 1.0; col += 1 }
      if (regime == "trend")                       { zMat(i, col) = t.toDouble; col += 1 }
      for (lag <- 1 to lags; j <- 0 until k) {
        zMat(i, col) = dY(t - lag, j); col += 1
      }
    }

    val (m0, m1) = if (nAux > 0) {
      val ztZInv = inv(zMat.t * zMat)
      val mz0    = r0mat - zMat * (ztZInv * zMat.t * r0mat)
      val mz1    = r1mat - zMat * (ztZInv * zMat.t * r1mat)
      (mz0, mz1)
    } else {
      (r0mat, r1mat)
    }

    val s00 = (m0.t * m0) / nObs.toDouble
    val s11 = (m1.t * m1) / nObs.toDouble
    val s01 = (m0.t * m1) / nObs.toDouble

    val s00Inv = inv(s00)
    val s11Inv = inv(s11)
    val aMat   = s11Inv * s01.t * s00Inv * s01

    val eig     = eigSym((aMat + aMat.t) / 2.0)
    val lambdas = eig.eigenvalues.toArray.sorted.reverse
      .map(l => math.max(l, 0.0))

    val traceStats: Array[Double] = (0 until k).map { r =>
      -nObs.toDouble * lambdas.drop(r).map(l =>
        math.log(1.0 - math.min(l, 0.999))).sum
    }.toArray

    val maxStats: Array[Double] = (0 until k).map { r =>
      -nObs.toDouble * math.log(1.0 - math.min(lambdas(r), 0.999))
    }.toArray

    JohansenResult(
      eigenvalues     = lambdas,
      traceStatistics = traceStats,
      maxStatistics   = maxStats,
      k               = k,
      nObs            = nObs,
      lags            = lags,
      regime          = regime
    )
  }

  def johansen(
                series: Array[Array[Double]],
                lags:   Int,
                regime: String
              ): JohansenResult = {
    require(series.nonEmpty, "Must provide at least one series")
    val n = series.head.length
    require(series.forall(_.length == n), "All series must have equal length")
    val yMat = DenseMatrix.tabulate(n, series.length)((i, j) => series(j)(i))
    johansen(yMat, lags, regime)
  }
}

case class EngleGrangerResult(
                               statistic:       Double,
                               lagOrder:        Int,
                               alpha:           Double,
                               beta:            Double,
                               residuals:       Array[Double],
                               criticalValue1:  Double,
                               criticalValue5:  Double,
                               criticalValue10: Double
                             ) {
  def rejectAt(level: Double = 0.05): Boolean = level match {
    case l if l <= 0.01 => statistic < criticalValue1
    case l if l <= 0.05 => statistic < criticalValue5
    case _              => statistic < criticalValue10
  }

  override def toString: String =
    s"""Engle-Granger Cointegration Test:
       |  t-statistic  = ${f"$statistic%.4f"}
       |  lag order    = $lagOrder
       |  alpha (int.) = ${f"$alpha%.4f"}
       |  beta (slope) = ${f"$beta%.4f"}
       |  CV 1%        = ${f"$criticalValue1%.3f"}
       |  CV 5%        = ${f"$criticalValue5%.3f"}
       |  CV 10%       = ${f"$criticalValue10%.3f"}
       |  ${if (rejectAt(0.05)) "Reject H0 (no cointegration) at 5%" else "Fail to reject H0 (no cointegration) at 5%"}
       |""".stripMargin
}

case class JohansenResult(
                           eigenvalues:     Array[Double],
                           traceStatistics: Array[Double],
                           maxStatistics:   Array[Double],
                           k:               Int,
                           nObs:            Int,
                           lags:            Int,
                           regime:          String
                         ) {
  override def toString: String = {
    val rows = (0 until k).map { r =>
      f"  r <= $r%d  trace = ${traceStatistics(r)}%8.3f  " +
        f"max-eig = ${maxStatistics(r)}%8.3f  lambda = ${eigenvalues(r)}%8.4f"
    }.mkString("\n")

    s"""Johansen Cointegration Test ($regime, lags=$lags):
       |  n = $nObs, k = $k
       |
       |${"  r".padTo(7, ' ')}${"trace".padTo(16, ' ')}${"max-eig".padTo(16, ' ')}lambda
       |$rows
       |""".stripMargin
  }
}