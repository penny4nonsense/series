package series.stats

import breeze.linalg._

object UnitRootTests {

  private val adfCriticalValues: Map[String, Array[Double]] = Map(
    "none"  -> Array(-2.565, -1.941, -1.616),
    "drift" -> Array(-3.430, -2.861, -2.567),
    "trend" -> Array(-3.960, -3.411, -3.127)
  )

  private val kpssCriticalValues: Map[String, Array[Double]] = Map(
    "level" -> Array(0.347, 0.463, 0.739),
    "trend" -> Array(0.119, 0.146, 0.216)
  )

  def adf(
           y:       Array[Double],
           maxLags: Int    = -1,
           regime:  String = "drift"
         ): ADFResult = {
    require(Seq("none", "drift", "trend").contains(regime),
      "regime must be 'none', 'drift', or 'trend'")

    val n      = y.length
    val lags   = if (maxLags < 0) autoLags(n) else maxLags
    val optLag = selectLagAIC(y, lags, regime)

    val (xMat, dy) = buildADFRegression(y, optLag, regime)
    val beta       = inv(xMat.t * xMat) * xMat.t * dy
    val resid      = dy - xMat * beta
    val n2         = dy.length
    val k          = xMat.cols
    val s2         = (resid.t * resid) / (n2 - k)
    val xTxInv     = inv(xMat.t * xMat)

    val gammaIdx = regime match {
      case "none"  => 0
      case "drift" => 1
      case "trend" => 2
    }
    val seGamma = math.sqrt(s2 * xTxInv(gammaIdx, gammaIdx))
    val tStat   = beta(gammaIdx) / seGamma
    val cv      = adfCriticalValues(regime)

    ADFResult(
      statistic       = tStat,
      lagOrder        = optLag,
      regime          = regime,
      criticalValue1  = cv(0),
      criticalValue5  = cv(1),
      criticalValue10 = cv(2)
    )
  }

  def pp(
          y:      Array[Double],
          lags:   Int    = -1,
          regime: String = "drift"
        ): PPResult = {
    require(Seq("none", "drift", "trend").contains(regime),
      "regime must be 'none', 'drift', or 'trend'")

    val n              = y.length
    val bw             = if (lags < 0) ppBandwidth(n) else lags
    val (xMat, dy)     = buildADFRegression(y, 0, regime)
    val beta           = inv(xMat.t * xMat) * xMat.t * dy
    val resid          = dy - xMat * beta
    val n2             = dy.length
    val k              = xMat.cols
    val s2             = (resid.t * resid) / (n2 - k)
    val lrv            = neweyWest(resid.toArray, bw)

    val gammaIdx = regime match {
      case "none"  => 0
      case "drift" => 1
      case "trend" => 2
    }

    val xTxInv   = inv(xMat.t * xMat)
    val seGamma  = math.sqrt(s2 * xTxInv(gammaIdx, gammaIdx))
    val tOLS     = beta(gammaIdx) / seGamma
    val correction = (lrv - s2) / (2.0 * math.sqrt(lrv) *
      math.sqrt(xTxInv(gammaIdx, gammaIdx)) *
      math.sqrt(n2.toDouble))
    val tStat    = math.sqrt(s2 / lrv) * tOLS - correction
    val cv       = adfCriticalValues(regime)

    PPResult(
      statistic       = tStat,
      bandwidth       = bw,
      regime          = regime,
      criticalValue1  = cv(0),
      criticalValue5  = cv(1),
      criticalValue10 = cv(2)
    )
  }

  def kpss(
            y:      Array[Double],
            lags:   Int    = -1,
            regime: String = "level"
          ): KPSSResult = {
    require(Seq("level", "trend").contains(regime),
      "regime must be 'level' or 'trend'")

    val n  = y.length
    val bw = if (lags < 0) ppBandwidth(n) else lags

    val resid = regime match {
      case "level" =>
        val mu = y.sum / n
        y.map(_ - mu)
      case "trend" =>
        val xMat = DenseMatrix.tabulate(n, 2)((i, j) => if (j == 0) 1.0 else i.toDouble)
        val yv   = DenseVector(y)
        val b    = inv(xMat.t * xMat) * xMat.t * yv
        (yv - xMat * b).toArray
    }

    val lrv  = neweyWest(resid, bw)
    val s    = resid.scanLeft(0.0)(_ + _).tail
    val stat = s.map(si => si * si).sum / (n.toDouble * n.toDouble * lrv)
    val cv   = kpssCriticalValues(regime)

    KPSSResult(
      statistic       = stat,
      bandwidth       = bw,
      regime          = regime,
      criticalValue10 = cv(0),
      criticalValue5  = cv(1),
      criticalValue1  = cv(2)
    )
  }

  private def neweyWest(resid: Array[Double], bw: Int): Double = {
    val n      = resid.length
    val mu     = resid.sum / n
    val v      = resid.map(_ - mu)
    val gamma0 = v.map(x => x * x).sum / n
    val acovSum = (1 to bw).map { j =>
      val w    = 1.0 - j.toDouble / (bw + 1)
      val acov = (0 until n - j).map(i => v(i) * v(i + j)).sum / n
      2.0 * w * acov
    }.sum
    gamma0 + acovSum
  }

  private def selectLagAIC(
                            y:      Array[Double],
                            maxLag: Int,
                            regime: String
                          ): Int = {
    (0 to maxLag).map { lag =>
      val (xMat, dy) = buildADFRegression(y, lag, regime)
      val beta       = inv(xMat.t * xMat) * xMat.t * dy
      val resid      = dy - xMat * beta
      val n          = dy.length
      val k          = xMat.cols
      val s2         = (resid.t * resid) / n
      val aic        = n * math.log(s2) + 2 * k
      (lag, aic)
    }.minBy(_._2)._1
  }

  private def buildADFRegression(
                                  y:      Array[Double],
                                  lags:   Int,
                                  regime: String
                                ): (DenseMatrix[Double], DenseVector[Double]) = {
    val n     = y.length
    val dy    = y.zip(y.tail).map { case (a, b) => b - a }
    val nReg  = n - 1 - lags
    val dyReg = DenseVector(dy.drop(lags))

    val nCols = regime match {
      case "none"  => 1 + lags
      case "drift" => 2 + lags
      case "trend" => 3 + lags
    }

    val xMat = DenseMatrix.zeros[Double](nReg, nCols)

    for (i <- 0 until nReg) {
      val t = i + lags + 1
      var col = 0
      if (regime == "drift" || regime == "trend") { xMat(i, col) = 1.0; col += 1 }
      if (regime == "trend")                       { xMat(i, col) = t.toDouble; col += 1 }
      xMat(i, col) = y(t - 1); col += 1
      for (j <- 1 to lags) { xMat(i, col) = dy(t - 1 - j); col += 1 }
    }

    (xMat, dyReg)
  }

  private def autoLags(n: Int): Int =
    math.floor(4.0 * math.pow(n / 100.0, 0.25)).toInt.max(1)

  private def ppBandwidth(n: Int): Int =
    math.floor(4.0 * math.pow(n / 100.0, 0.25)).toInt.max(1)
}

case class ADFResult(
                      statistic:       Double,
                      lagOrder:        Int,
                      regime:          String,
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
    s"""ADF Test ($regime):
       |  t-statistic = ${f"$statistic%.4f"}
       |  lag order   = $lagOrder
       |  CV 1%       = ${f"$criticalValue1%.3f"}
       |  CV 5%       = ${f"$criticalValue5%.3f"}
       |  CV 10%      = ${f"$criticalValue10%.3f"}
       |  ${if (rejectAt(0.05)) "Reject H0 (unit root) at 5%" else "Fail to reject H0 (unit root) at 5%"}
       |""".stripMargin
}

case class PPResult(
                     statistic:       Double,
                     bandwidth:       Int,
                     regime:          String,
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
    s"""PP Test ($regime):
       |  t-statistic = ${f"$statistic%.4f"}
       |  bandwidth   = $bandwidth
       |  CV 1%       = ${f"$criticalValue1%.3f"}
       |  CV 5%       = ${f"$criticalValue5%.3f"}
       |  CV 10%      = ${f"$criticalValue10%.3f"}
       |  ${if (rejectAt(0.05)) "Reject H0 (unit root) at 5%" else "Fail to reject H0 (unit root) at 5%"}
       |""".stripMargin
}

case class KPSSResult(
                       statistic:       Double,
                       bandwidth:       Int,
                       regime:          String,
                       criticalValue10: Double,
                       criticalValue5:  Double,
                       criticalValue1:  Double
                     ) {
  def rejectAt(level: Double = 0.05): Boolean = level match {
    case l if l <= 0.01 => statistic > criticalValue1
    case l if l <= 0.05 => statistic > criticalValue5
    case _              => statistic > criticalValue10
  }

  override def toString: String =
    s"""KPSS Test ($regime):
       |  statistic = ${f"$statistic%.4f"}
       |  bandwidth = $bandwidth
       |  CV 1%     = ${f"$criticalValue1%.3f"}
       |  CV 5%     = ${f"$criticalValue5%.3f"}
       |  CV 10%    = ${f"$criticalValue10%.3f"}
       |  ${if (rejectAt(0.05)) "Reject H0 (stationarity) at 5%" else "Fail to reject H0 (stationarity) at 5%"}
       |""".stripMargin
}