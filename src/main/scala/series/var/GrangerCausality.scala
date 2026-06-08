package series.`var`

import breeze.linalg._
import breeze.stats.distributions.FDistribution

/**
 * Granger causality tests for VAR models.
 *
 * X Granger-causes Y if lagged X improves prediction of Y
 * beyond what lagged Y alone provides.
 *
 * Implemented as an F-test on the joint significance of
 * lagged coefficients of X in the equation for Y.
 *
 * H0: X does not Granger-cause Y (all lag coefficients of X in Y equation = 0)
 * H1: X Granger-causes Y
 */
object GrangerCausality {

  /**
   * Tests whether series `cause` Granger-causes series `effect`.
   *
   * @param fit    fitted VarFit
   * @param cause  index of causing series (0-based)
   * @param effect index of effect series (0-based)
   * @return GrangerResult with F-statistic and p-value
   */
  def test(
            fit: VarFit,
            cause: Int,
            effect: Int
          ): GrangerResult = {
    require(cause != effect, "cause and effect must be different series")
    require(cause >= 0 && cause < fit.k, s"cause index out of range")
    require(effect >= 0 && effect < fit.k, s"effect index out of range")

    val p = fit.p
    val k = fit.k

    val Y = fit.Y
    val model = VarModel(p, k, fit.includeDrift)
    val Z = model.buildRegressors(Y)
    val nObs = Z.rows // actual regressor rows, not fit.n
    val nCoeffs = Z.cols

    val offset = if (fit.includeDrift) 1 else 0
    val causeColIndices = (1 to p).map(lag => offset + (lag - 1) * k + cause)

    val keepCols = (0 until nCoeffs).filterNot(causeColIndices.contains)
    val Zr = DenseMatrix.tabulate(nObs, keepCols.length)((i, j) => Z(i, keepCols(j)))

    val Yresp = Y(p until Y.rows, ::)
    val yEffect = Yresp(::, effect)

    val buOLS = inv(Z.t * Z) * Z.t * yEffect
    val residU = yEffect - Z * buOLS
    val ssrU = residU.t * residU

    val brOLS = inv(Zr.t * Zr) * Zr.t * yEffect
    val residR = yEffect - Zr * brOLS
    val ssrR = residR.t * residR

    val q = p
    val dfU = nObs - nCoeffs
    val fStat = ((ssrR - ssrU) / q) / (ssrU / dfU)
    val pValue = 1.0 - new FDistribution(q, dfU).cdf(fStat)

    GrangerResult(
      fStatistic = fStat,
      dfNum = q,
      dfDen = dfU,
      pValue = pValue,
      cause = cause,
      effect = effect,
      p = p
    )
  }

  /**
   * Tests all pairwise Granger causality relationships.
   *
   * @param fit fitted VarFit
   * @return matrix of GrangerResults (k x k), diagonal is None
   */
  def testAll(fit: VarFit): Array[Array[Option[GrangerResult]]] = {
    Array.tabulate(fit.k, fit.k) { (effect, cause) =>
      if (cause == effect) None
      else Some(test(fit, cause, effect))
    }
  }

  /**
   * Prints a readable causality matrix.
   */
  def causalityMatrix(fit: VarFit, alpha: Double = 0.05): String = {
    val results = testAll(fit)
    val sb      = new StringBuilder
    sb.append(s"Granger Causality Matrix (alpha=$alpha):\n")
    sb.append(s"  Row = effect, Col = cause\n")
    sb.append(s"  * = significant at $alpha\n\n")

    val header = (0 until fit.k).map(j => s"  y${j+1}").mkString("       ", "", "")
    sb.append(header + "\n")

    for (effect <- 0 until fit.k) {
      sb.append(f"  y${effect+1}%4s  ")
      for (cause <- 0 until fit.k) {
        if (cause == effect) sb.append(f"  ${"—"}%5s")
        else {
          val r   = results(effect)(cause).get
          val sig = if (r.pValue < alpha) "*" else " "
          sb.append(f"${r.pValue}%5.3f$sig")
        }
      }
      sb.append("\n")
    }
    sb.toString
  }
}

case class GrangerResult(
                          fStatistic: Double,
                          dfNum:      Int,
                          dfDen:      Int,
                          pValue:     Double,
                          cause:      Int,
                          effect:     Int,
                          p:          Int
                        ) {
  def isSignificant(alpha: Double = 0.05): Boolean = pValue < alpha

  override def toString: String =
    s"""Granger Causality Test (y${cause+1} -> y${effect+1}):
       |  F($dfNum, $dfDen) = ${f"$fStatistic%.4f"}
       |  p-value   = ${f"$pValue%.4f"}
       |  ${if (isSignificant()) s"y${cause+1} Granger-causes y${effect+1} at 5%" else s"No evidence that y${cause+1} Granger-causes y${effect+1}"}
       |""".stripMargin
}