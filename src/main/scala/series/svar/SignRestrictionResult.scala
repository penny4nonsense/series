package series.svar

import breeze.linalg._

/**
 * Results from sign-restriction SVAR identification.
 *
 * Set-identified — reports distribution of admissible IRFs
 * rather than a point estimate. Summarized by median and
 * 16th/84th percentile bands (one posterior standard deviation)
 * following Uhlig (2005) convention.
 *
 * @param admissibleIRFs   array of admissible IRF sets
 * @param admissibleB0invs array of admissible impact matrices
 * @param nDraws           total draws attempted
 * @param nAccepted        draws satisfying restrictions
 * @param h                IRF horizon
 * @param k                number of series
 * @param restrictions     sign restrictions imposed
 * @param shockNames       names for structural shocks
 */
case class SignRestrictionResult(
                                  admissibleIRFs:   Array[Array[DenseMatrix[Double]]],
                                  admissibleB0invs: Array[DenseMatrix[Double]],
                                  nDraws:           Int,
                                  nAccepted:        Int,
                                  h:                Int,
                                  k:                Int,
                                  restrictions:     Seq[SignRestrictions.SignRestriction],
                                  shockNames:       Array[String]
                                ) {
  /** Acceptance rate */
  def acceptanceRate: Double = nAccepted.toDouble / nDraws

  /**
   * Extracts the distribution of IRFs for a given series/shock pair.
   * Returns array of length nAccepted, each entry an IRF path.
   *
   * @param from shock index
   * @param to   series index
   * @return array of IRF paths (nAccepted x h+1)
   */
  def irfDistribution(from: Int, to: Int): Array[Array[Double]] =
    admissibleIRFs.map(sirfs => sirfs.map(_(to, from)))

  /**
   * Median IRF for a given series/shock pair.
   */
  def medianIRF(from: Int, to: Int): Array[Double] = {
    val dist = irfDistribution(from, to)
    (0 to h).map { s =>
      val vals = dist.map(_(s)).sorted
      vals(vals.length / 2)
    }.toArray
  }

  /**
   * Percentile IRF bands.
   *
   * @param from       shock index
   * @param to         series index
   * @param lowerPerc  lower percentile (e.g. 0.16)
   * @param upperPerc  upper percentile (e.g. 0.84)
   * @return (lower, upper) band arrays
   */
  def irfBands(
                from:       Int,
                to:         Int,
                lowerPerc:  Double = 0.16,
                upperPerc:  Double = 0.84
              ): (Array[Double], Array[Double]) = {
    val dist = irfDistribution(from, to)
    val lower = (0 to h).map { s =>
      val vals = dist.map(_(s)).sorted
      vals((vals.length * lowerPerc).toInt)
    }.toArray
    val upper = (0 to h).map { s =>
      val vals = dist.map(_(s)).sorted
      vals((vals.length * upperPerc).toInt)
    }.toArray
    (lower, upper)
  }

  /**
   * Forecast error variance decomposition.
   * Median FEVD across admissible models.
   */
  def fevd: Array[Array[Array[Double]]] = {
    // Compute FEVD for each admissible model
    val allFevds = admissibleIRFs.map { sirfs =>
      val contributions = Array.tabulate(h+1, k, k) { (s, i, j) =>
        sirfs(s)(i, j) * sirfs(s)(i, j)
      }
      val cumulative = Array.tabulate(h+1, k, k) { (s, i, j) =>
        (0 to s).map(t => contributions(t)(i)(j)).sum
      }
      Array.tabulate(h+1, k, k) { (s, i, j) =>
        val total = (0 until k).map(jj => cumulative(s)(i)(jj)).sum
        if (total > 0) cumulative(s)(i)(j) / total else 0.0
      }
    }

    // Median across draws
    Array.tabulate(h+1, k, k) { (s, i, j) =>
      val vals = allFevds.map(_(s)(i)(j)).sorted
      vals(vals.length / 2)
    }
  }

  override def toString: String = {
    val sb = new StringBuilder
    sb.append(s"Sign Restriction SVAR Result\n")
    sb.append(s"  k = $k, h = $h\n")
    sb.append(f"  Acceptance rate = $acceptanceRate%.3f " +
      s"($nAccepted/$nDraws)\n\n")

    sb.append("  Restrictions:\n")
    restrictions.foreach { r =>
      val signStr = if (r.sign == 1) "positive" else "negative"
      sb.append(s"    IRF(y${r.series+1}, ${shockNames(r.shock)}, " +
        s"h=${r.horizon}) >= 0 (${signStr})\n")
    }
    sb.append("\n")

    for (j <- 0 until k) {
      sb.append(s"  Median IRF — ${shockNames(j)} " +
        s"[16th, median, 84th percentile]:\n")
      sb.append(f"  ${"h"}%5s" +
        (0 until k).map(i => f"  ${s"y${i+1}"}%20s").mkString + "\n")
      for (s <- 0 to math.min(h, 20)) {
        sb.append(f"  $s%5d")
        for (i <- 0 until k) {
          val med          = medianIRF(j, i)(s)
          val (low, high)  = irfBands(j, i)
          sb.append(f"  [${low(s)}%6.3f, $med%6.3f, ${high(s)}%6.3f]")
        }
        sb.append("\n")
      }
      sb.append("\n")
    }
    sb.toString
  }
}