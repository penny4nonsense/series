package series.`var`

import breeze.linalg._

/**
 * Lag order selection for VAR models.
 *
 * Fits VAR(p) for p = 1..maxLag and selects the order
 * minimizing AIC, BIC, or HQIC.
 *
 * AIC  tends to overselect — good for forecasting
 * BIC  consistent — good for inference
 * HQIC intermediate between AIC and BIC
 */
object VarLagSelection {

  case class LagSelectionResult(
                                 criteria: Map[Int, (Double, Double, Double)],  // lag -> (AIC, BIC, HQIC)
                                 aicLag:   Int,
                                 bicLag:   Int,
                                 hqicLag:  Int
                               ) {
    override def toString: String = {
      val header = f"  ${"lag"}%5s  ${"AIC"}%12s  ${"BIC"}%12s  ${"HQIC"}%12s"
      val rows = criteria.toSeq.sortBy(_._1).map { case (lag, (aic, bic, hqic)) =>
        val aicMark  = if (lag == aicLag)  "*" else " "
        val bicMark  = if (lag == bicLag)  "*" else " "
        val hqicMark = if (lag == hqicLag) "*" else " "
        f"  $lag%5d  $aic%11.4f$aicMark  $bic%11.4f$bicMark  $hqic%11.4f$hqicMark"
      }.mkString("\n")
      s"VAR Lag Selection:\n$header\n$rows\n\n* = selected"
    }
  }

  /**
   * Selects VAR lag order by information criteria.
   *
   * @param Y      data matrix (n x k)
   * @param maxLag maximum lag to consider
   * @param includeDrift whether to include constant
   * @return LagSelectionResult with criteria values and selected lags
   */
  def select(
              Y:            DenseMatrix[Double],
              maxLag:       Int,
              includeDrift: Boolean = true
            ): LagSelectionResult = {
    require(maxLag > 0, "maxLag must be positive")

    val criteria = (1 to maxLag).map { lag =>
      val model = VarModel(lag, Y.cols, includeDrift)
      val fit   = model.fit(Y)
      lag -> (fit.aic, fit.bic, fit.hqic)
    }.toMap

    val aicLag  = criteria.minBy(_._2._1)._1
    val bicLag  = criteria.minBy(_._2._2)._1
    val hqicLag = criteria.minBy(_._2._3)._1

    LagSelectionResult(criteria, aicLag, bicLag, hqicLag)
  }
}