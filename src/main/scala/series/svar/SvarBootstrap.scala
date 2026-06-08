package series.svar

import breeze.linalg._
import series.`var`.{VarModel, VarFit}

/**
 * Bootstrap confidence bands for point-identified SVAR IRFs.
 *
 * Uses the residual bootstrap — resamples reduced form residuals
 * with replacement, reconstructs the data, re-estimates the VAR,
 * re-identifies the structural model, and collects the distribution
 * of bootstrapped IRFs.
 *
 * Two bootstrap schemes:
 *   Standard  — resamples residuals iid
 *   Wild      — multiplies residuals by random signs (Rademacher)
 *               better for heteroskedastic errors
 */
object SvarBootstrap {

  sealed trait BootstrapScheme
  case object Standard extends BootstrapScheme
  case object Wild     extends BootstrapScheme

  /**
   * Bootstraps confidence bands for Cholesky-identified SVAR.
   *
   * @param model      SvarModel
   * @param h          IRF horizon
   * @param nBoot      number of bootstrap replications
   * @param scheme     bootstrap scheme
   * @param seed       random seed
   * @param shockNames optional shock names
   * @return BootstrapResult with IRF distribution
   */
  def bootstrapCholesky(
                         model:      SvarModel,
                         h:          Int,
                         nBoot:      Int    = 1000,
                         scheme:     BootstrapScheme = Standard,
                         seed:       Long   = 42,
                         shockNames: Option[Array[String]] = None
                       ): BootstrapResult = {
    val rng   = new scala.util.Random(seed)
    val fit   = model.fit
    val k     = fit.k
    val p     = fit.p
    val Y     = fit.Y
    val n     = Y.rows
    val nObs  = fit.residuals.rows

    val bootIRFs = scala.collection.mutable.ArrayBuffer[Array[DenseMatrix[Double]]]()

    var b = 0
    while (b < nBoot) {
      try {
        // Resample residuals
        val bootResid = DenseMatrix.zeros[Double](nObs, k)
        for (t <- 0 until nObs) {
          val srcIdx = rng.nextInt(nObs)
          val sign   = scheme match {
            case Standard => 1.0
            case Wild     => if (rng.nextBoolean()) 1.0 else -1.0
          }
          for (j <- 0 until k)
            bootResid(t, j) = sign * fit.residuals(srcIdx, j)
        }

        // Reconstruct bootstrap data
        val Yboot = DenseMatrix.zeros[Double](n, k)
        // Initialize with first p observations
        for (t <- 0 until p; j <- 0 until k)
          Yboot(t, j) = Y(t, j)

        // Recurse forward
        for (t <- p until n) {
          val tResid = t - p
          var yhat   = if (fit.includeDrift) fit.intercept.copy
          else DenseVector.zeros[Double](k)
          for (lag <- 1 to p) {
            val A    = fit.lagMatrix(lag)
            val ylag = Yboot(t - lag, ::).t
            yhat += A * ylag
          }
          for (j <- 0 until k)
            Yboot(t, j) = yhat(j) + bootResid(tResid, j)
        }

        // Re-estimate VAR
        val bootFit   = VarModel(p, k, fit.includeDrift).fit(Yboot)
        val bootSvar  = new SvarModel(bootFit)
        val bootResult = ShortRunIdentification.cholesky(bootSvar, h)

        bootIRFs += bootResult.sirfs
        b += 1
      } catch {
        case _: Exception => // skip failed bootstrap replications
      }
    }

    BootstrapResult(
      bootIRFs   = bootIRFs.toArray,
      nBoot      = nBoot,
      h          = h,
      k          = k,
      shockNames = shockNames.getOrElse(
        (0 until k).map(j => s"shock${j+1}").toArray
      )
    )
  }

  /**
   * Bootstraps confidence bands for Blanchard-Quah SVAR.
   */
  def bootstrapBlanchardQuah(
                              model:      SvarModel,
                              h:          Int,
                              nBoot:      Int    = 1000,
                              scheme:     BootstrapScheme = Standard,
                              seed:       Long   = 42,
                              shockNames: Option[Array[String]] = None
                            ): BootstrapResult = {
    val rng  = new scala.util.Random(seed)
    val fit  = model.fit
    val k    = fit.k
    val p    = fit.p
    val Y    = fit.Y
    val n    = Y.rows
    val nObs = fit.residuals.rows

    val bootIRFs = scala.collection.mutable.ArrayBuffer[Array[DenseMatrix[Double]]]()

    var b = 0
    while (b < nBoot) {
      try {
        val bootResid = DenseMatrix.zeros[Double](nObs, k)
        for (t <- 0 until nObs) {
          val srcIdx = rng.nextInt(nObs)
          val sign   = scheme match {
            case Standard => 1.0
            case Wild     => if (rng.nextBoolean()) 1.0 else -1.0
          }
          for (j <- 0 until k)
            bootResid(t, j) = sign * fit.residuals(srcIdx, j)
        }

        val Yboot = DenseMatrix.zeros[Double](n, k)
        for (t <- 0 until p; j <- 0 until k)
          Yboot(t, j) = Y(t, j)

        for (t <- p until n) {
          val tResid = t - p
          var yhat   = if (fit.includeDrift) fit.intercept.copy
          else DenseVector.zeros[Double](k)
          for (lag <- 1 to p) {
            val A    = fit.lagMatrix(lag)
            val ylag = Yboot(t - lag, ::).t
            yhat += A * ylag
          }
          for (j <- 0 until k)
            Yboot(t, j) = yhat(j) + bootResid(tResid, j)
        }

        val bootFit    = VarModel(p, k, fit.includeDrift).fit(Yboot)
        val bootSvar   = new SvarModel(bootFit)
        val bootResult = LongRunIdentification.blanchardQuah(bootSvar, h)

        bootIRFs += bootResult.sirfs
        b += 1
      } catch {
        case _: Exception =>
      }
    }

    BootstrapResult(
      bootIRFs   = bootIRFs.toArray,
      nBoot      = nBoot,
      h          = h,
      k          = k,
      shockNames = shockNames.getOrElse(
        (0 until k).map(j => s"shock${j+1}").toArray
      )
    )
  }
}

/**
 * Bootstrap IRF confidence band results.
 *
 * @param bootIRFs   array of bootstrapped IRF sets (nBoot x h+1 x k x k)
 * @param nBoot      number of bootstrap replications
 * @param h          IRF horizon
 * @param k          number of series
 * @param shockNames shock names
 */
case class BootstrapResult(
                            bootIRFs:   Array[Array[DenseMatrix[Double]]],
                            nBoot:      Int,
                            h:          Int,
                            k:          Int,
                            shockNames: Array[String]
                          ) {
  /**
   * Percentile confidence band for IRF of series `to` to shock `from`.
   *
   * @param from      shock index
   * @param to        series index
   * @param lowerPerc lower percentile (e.g. 0.05 for 90% band)
   * @param upperPerc upper percentile (e.g. 0.95 for 90% band)
   * @return (lower, upper) confidence band arrays of length h+1
   */
  def confidenceBand(
                      from:       Int,
                      to:         Int,
                      lowerPerc:  Double = 0.05,
                      upperPerc:  Double = 0.95
                    ): (Array[Double], Array[Double]) = {
    val lower = (0 to h).map { s =>
      val vals = bootIRFs.map(_(s)(to, from)).sorted
      vals((vals.length * lowerPerc).toInt)
    }.toArray
    val upper = (0 to h).map { s =>
      val vals = bootIRFs.map(_(s)(to, from)).sorted
      vals((vals.length * upperPerc).toInt)
    }.toArray
    (lower, upper)
  }

  /**
   * Median IRF across bootstrap replications.
   */
  def medianIRF(from: Int, to: Int): Array[Double] =
    (0 to h).map { s =>
      val vals = bootIRFs.map(_(s)(to, from)).sorted
      vals(vals.length / 2)
    }.toArray

  override def toString: String = {
    val sb = new StringBuilder
    sb.append(s"Bootstrap SVAR Confidence Bands\n")
    sb.append(s"  nBoot = $nBoot, h = $h, k = $k\n\n")
    for (j <- 0 until k) {
      sb.append(s"  Shock: ${shockNames(j)}\n")
      sb.append(f"  ${"h"}%5s" +
        (0 until k).map(i => f"  ${s"y${i+1}[5,50,95]"}%22s").mkString + "\n")
      for (s <- 0 to math.min(h, 20)) {
        sb.append(f"  $s%5d")
        for (i <- 0 until k) {
          val (lo, hi) = confidenceBand(j, i)
          val med      = medianIRF(j, i)(s)
          sb.append(f"  [${lo(s)}%6.3f,$med%6.3f,${hi(s)}%6.3f]")
        }
        sb.append("\n")
      }
      sb.append("\n")
    }
    sb.toString
  }
}