package series.svar

import breeze.linalg._
import series.`var`.VarFit

/**
 * Results from point-identified SVAR estimation.
 * Used for Cholesky and zero restriction identification.
 *
 * @param B0inv          impact matrix B_0^{-1} (k x k)
 * @param sirfs          structural IRFs (h+1 x k x k)
 * @param shocks         structural shock series (n-p x k)
 * @param fit            underlying reduced form VarFit
 * @param h              IRF horizon
 * @param identification description of identification scheme
 * @param shockNames     names for structural shocks
 */
case class SvarResult(
                       B0inv:          DenseMatrix[Double],
                       sirfs:          Array[DenseMatrix[Double]],
                       shocks:         DenseMatrix[Double],
                       fit:            VarFit,
                       h:              Int,
                       identification: String,
                       shockNames:     Array[String]
                     ) {
  val k: Int = B0inv.rows

  /**
   * IRF of series `to` to shock `from` over all horizons.
   */
  def irf(from: Int, to: Int): Array[Double] =
    sirfs.map(_(to, from))

  /**
   * Forecast error variance decomposition under structural identification.
   * Proportion of h-step forecast error variance of series i
   * due to shock j.
   */
  def fevd: Array[Array[Array[Double]]] = {
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

  override def toString: String = {
    val sb = new StringBuilder
    sb.append(s"SVAR Result — $identification\n")
    sb.append(s"  k = $k series, h = $h horizon\n\n")

    sb.append("  Impact matrix B0inv:\n")
    for (i <- 0 until k) {
      sb.append("  ")
      for (j <- 0 until k)
        sb.append(f"${B0inv(i,j)}%10.4f")
      sb.append("\n")
    }
    sb.append("\n")

    for (j <- 0 until k) {
      sb.append(s"  Structural IRF — ${shockNames(j)}:\n")
      sb.append(f"  ${"h"}%5s" +
        (0 until k).map(i => f"  ${s"y${i+1}"}%9s").mkString + "\n")
      for (s <- 0 to math.min(h, 20))
        sb.append(f"  $s%5d" +
          (0 until k).map(i => f"  ${sirfs(s)(i,j)}%9.4f").mkString + "\n")
      sb.append("\n")
    }
    sb.toString
  }
}