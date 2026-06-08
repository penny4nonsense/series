package series.vecm

import breeze.linalg._

case class VecmModel(
  k:            Int,
  p:            Int,
  r:            Int,
  includeDrift: Boolean = true
) {
  require(k > 1, "VECM requires at least 2 series")
  require(p > 0, "lag order must be positive")
  require(r > 0, "cointegrating rank must be positive")
  require(r < k, "cointegrating rank must be less than k")

  def fit(Y: DenseMatrix[Double]): VecmFit = {
    require(Y.cols == k, s"Expected $k series, got ${Y.cols}")
    require(Y.rows > p + k, "Insufficient observations")

    val n    = Y.rows
    val nObs = n - p

    val dY = DenseMatrix.tabulate(n-1, k)((t, j) => Y(t+1, j) - Y(t, j))

    val r0 = dY(p-1 until n-1, ::).copy
    val r1 = Y(p-1 until n-1, ::).copy

    val nLags = p - 1
    val nDet  = if (includeDrift) 1 else 0
    val nAux  = k * nLags + nDet

    val zOpt = if (nAux > 0) {
      val zmat = DenseMatrix.zeros[Double](nObs, nAux)
      for (t <- 0 until nObs) {
        var col = 0
        if (includeDrift) { zmat(t, col) = 1.0; col += 1 }
        for (lag <- 1 to nLags; j <- 0 until k) {
          zmat(t, col) = dY(t + p - 1 - lag, j); col += 1
        }
      }
      Some(zmat)
    } else None

    val (m0, m1) = zOpt match {
      case Some(zmat) =>
        val ztZInv = inv(zmat.t * zmat)
        val ztR0   = zmat.t * r0
        val ztR1   = zmat.t * r1
        val mz0    = r0 - zmat * (ztZInv * ztR0)
        val mz1    = r1 - zmat * (ztZInv * ztR1)
        (mz0.copy, mz1.copy)
      case None => (r0, r1)
    }

    val s00 = (m0.t * m0) / nObs.toDouble
    val s11 = (m1.t * m1) / nObs.toDouble
    val s01 = (m0.t * m1) / nObs.toDouble

    val s00Inv = inv(s00)
    val s11Inv = inv(s11)
    val aMat   = s11Inv * s01.t * s00Inv * s01
    val aSymm  = ((aMat + aMat.t) / 2.0).copy

    val eigResult = eigSym(aSymm)

    val eigVals = eigResult.eigenvalues.toArray
    val eigVecs = eigResult.eigenvectors  // k x k, columns are eigenvectors

    // Sort descending by eigenvalue
    val sortedIdx = eigVals.zipWithIndex.sortBy(-_._1).map(_._2)
    val lambdas   = sortedIdx.map(i => math.max(eigVals(i), 0.0))
    val vecs      = sortedIdx.map(i => eigVecs(::, i).copy)

    val betaRaw = DenseMatrix.zeros[Double](k, r)
    for (i <- 0 until k; j <- 0 until r) betaRaw(i, j) = vecs(j)(i)
    val norm11     = betaRaw.t * s11 * betaRaw
    val normFactor = diag(norm11).map(x => 1.0 / math.sqrt(math.abs(x)))
    val beta = DenseMatrix.zeros[Double](k, r)
    for (i <- 0 until k; j <- 0 until r)
      beta(i, j) = betaRaw(i, j) * normFactor(j)

    val alpha = (s01 * beta * inv(beta.t * s11 * beta)).copy
    val pi    = (alpha * beta.t).copy

    val r0adj = (r0 - r1 * pi.t).copy

    val (gammas, intercept, sigma): (Array[DenseMatrix[Double]], DenseVector[Double], DenseMatrix[Double]) =
      zOpt match {
        case Some(zmat) =>
          val ztZInv    = inv(zmat.t * zmat)
          val bMat      = (ztZInv * (zmat.t * r0adj)).copy
          val residuals = (r0adj - zmat * bMat).copy
          val intVec    = if (includeDrift) bMat(0, ::).t.copy
                          else DenseVector.zeros[Double](k)
          val gammaList: Array[DenseMatrix[Double]] = (0 until nLags).map { lag =>
            val offset = if (includeDrift) 1 + lag * k else lag * k
            bMat(offset until offset + k, ::).t.copy
          }.toArray
          val sig = ((residuals.t * residuals) / (nObs - k).toDouble).copy
          (gammaList, intVec, sig)

        case None =>
          val sig = ((r0adj.t * r0adj) / nObs.toDouble).copy
          (Array.empty[DenseMatrix[Double]], DenseVector.zeros[Double](k), sig)
      }

    val finalResid: DenseMatrix[Double] = zOpt match {
      case Some(zmat) =>
        val ztZInv = inv(zmat.t * zmat)
        (r0adj - zmat * (ztZInv * (zmat.t * r0adj))).copy
      case None => r0adj
    }

    val ll = logLikelihood(finalResid, sigma, nObs)

    VecmFit(
      alpha        = alpha,
      beta         = beta,
      Pi           = pi,
      gammas       = gammas,
      intercept    = intercept,
      Sigma        = sigma,
      logLik       = ll,
      eigenvalues  = lambdas.take(r),
      n            = nObs,
      k            = k,
      p            = p,
      r            = r,
      Y            = Y,
      residuals    = finalResid,
      includeDrift = includeDrift
    )
  }

  private def logLikelihood(
    resid: DenseMatrix[Double],
    sigma: DenseMatrix[Double],
    n:     Int
  ): Double = {
    val logDet = math.log(math.abs(det(sigma)))
    val sigInv = inv(sigma)
    val tr = (0 until n).map { t =>
      val e = resid(t, ::).t.copy
      e.t * sigInv * e
    }.sum
    -0.5 * (n * k * math.log(2 * math.Pi) + n * logDet + tr)
  }
}
