package series.smoothing

/**
 * Automatic ETS model selection.
 *
 * Fits all 15 admissible ETS models to the data and ranks them
 * by the specified information criterion.
 *
 * Follows Hyndman & Khandakar (2008) "Automatic Time Series Forecasting"
 * Journal of Statistical Software.
 *
 * @param period     seasonal period (1 = non-seasonal, fits only 3 models)
 * @param criterion  selection criterion
 */
class AutoEts(
  period:    Int,
  criterion: AutoEts.Criterion = AutoEts.AICc
) {
  require(period >= 1, "period must be at least 1")

  /**
   * Fits all admissible models and returns results ranked by criterion.
   *
   * @param y time series
   * @return AutoEtsResult with all fits ranked
   */
  def fit(y: Array[Double]): AutoEtsResult = {
    val candidates = if (period == 1) {
      // Non-seasonal: only 6 models (no seasonal component)
      EtsSpec.allAdmissible(1).filter(_.seasonal == NoSeasonal)
    } else {
      EtsSpec.allAdmissible(period)
    }.filter { spec =>
      // Skip models requiring more data than available
      y.length >= spec.period * 2 + spec.nParams + 1
    }

    val fits = candidates.flatMap { spec =>
      try {
        val estimator = new EtsEstimator(spec)
        Some(estimator.fit(y))
      } catch {
        case _: Exception => None
      }
    }

    require(fits.nonEmpty, "No ETS models could be fit to this series")

    val scored = fits.map { f =>
      val score = criterion match {
        case AutoEts.AIC  => f.aic
        case AutoEts.AICc => f.aicc
        case AutoEts.BIC  => f.bic
      }
      (f, score)
    }.sortBy(_._2)

    AutoEtsResult(
      fits      = scored.map(_._1).toArray,
      scores    = scored.map(_._2).toArray,
      criterion = criterion,
      best      = scored.head._1
    )
  }
}

object AutoEts {
  sealed trait Criterion
  case object AIC  extends Criterion
  case object AICc extends Criterion
  case object BIC  extends Criterion
}

case class AutoEtsResult(
  fits:      Array[EtsFit],
  scores:    Array[Double],
  criterion: AutoEts.Criterion,
  best:      EtsFit
) {
  def nModels: Int = fits.length

  def ranking: String = {
    val sb = new StringBuilder
    sb.append(s"AutoETS Model Selection ($criterion):\n")
    sb.append(f"  ${"Model"}%-15s  ${"Score"}%10s\n")
    fits.zip(scores).foreach { case (f, s) =>
      val marker = if (f.spec.name == best.spec.name) " *" else ""
      sb.append(f"  ${f.spec.name}%-15s  $s%10.4f$marker\n")
    }
    sb.toString
  }

  override def toString: String =
    s"AutoETS: best model = ${best.spec.name}\n" + best.toString
}
