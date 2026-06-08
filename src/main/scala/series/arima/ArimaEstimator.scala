package series.arima

import breeze.linalg._
import breeze.optimize._
import series.kalman._

/**
 * Estimates ARIMA parameters via CSS and exact MLE.
 *
 * CSS (conditional sum of squares) provides fast approximate estimates
 * used as starting values for exact MLE via LBFGS.
 *
 * @param model ArimaModel specifying p, d, q
 */
class ArimaEstimator(model: ArimaModel) {

  private def unpack(params: DenseVector[Double]): (Array[Double], Array[Double], Double) = {
    val phi    = params(0 until model.p).toArray
    val theta  = params(model.p until model.p + model.q).toArray
    val sigma2 = math.exp(params(model.p + model.q))
    (phi, theta, sigma2)
  }

  private def pack(
                    phi:    Array[Double],
                    theta:  Array[Double],
                    sigma2: Double
                  ): DenseVector[Double] =
    DenseVector(phi ++ theta ++ Array(math.log(sigma2)))

  private def negLogLik(
                         params: DenseVector[Double],
                         obs:    Seq[DenseVector[Double]]
                       ): Double = {
    try {
      val (phi, theta, sigma2) = unpack(params)
      val kalmanModel = model.toKalmanModel(phi, theta, sigma2)
      val initial     = model.initialState(phi, theta, sigma2)
      -LogLikelihoodAccumulator.logLikelihood(initial, kalmanModel, obs)
    } catch {
      case _: Exception => Double.MaxValue
    }
  }

  private def negCssLogLik(
                            params: DenseVector[Double],
                            obs:    Seq[DenseVector[Double]]
                          ): Double = {
    try {
      val (phi, theta, sigma2) = unpack(params)
      val kalmanModel = model.toKalmanModel(phi, theta, sigma2)
      val initial     = model.initialState(phi, theta, sigma2)
      -LogLikelihoodAccumulator.cssLogLikelihood(initial, kalmanModel, obs)
    } catch {
      case _: Exception => Double.MaxValue
    }
  }

  private def numericalHessian(
                                params: DenseVector[Double],
                                obs:    Seq[DenseVector[Double]]
                              ): DenseMatrix[Double] = {
    val n = params.length
    val H = DenseMatrix.zeros[Double](n, n)
    val h = 1e-4
    for (i <- 0 until n; j <- 0 until n) {
      val pp = params.copy; pp(i) += h; pp(j) += h
      val pm = params.copy; pm(i) += h; pm(j) -= h
      val mp = params.copy; mp(i) -= h; mp(j) += h
      val mm = params.copy; mm(i) -= h; mm(j) -= h
      H(i, j) = (negLogLik(pp, obs) - negLogLik(pm, obs) -
        negLogLik(mp, obs) + negLogLik(mm, obs)) / (4 * h * h)
    }
    H
  }

  private def standardErrors(
                              params: DenseVector[Double],
                              obs:    Seq[DenseVector[Double]]
                            ): Array[Double] = {
    try {
      val H    = numericalHessian(params, obs)
      val Hinv = inv(H)
      (0 until params.length).map(i =>
        math.sqrt(math.abs(Hinv(i, i)))
      ).toArray
    } catch {
      case _: Exception => Array.fill(params.length)(Double.NaN)
    }
  }

  def fit(y: Array[Double], cssOnly: Boolean = false): ArimaFit = {
    val diffed = difference(y, model.d)
    val obs: Seq[DenseVector[Double]] = diffed.map(v => DenseVector(v)).toIndexedSeq

    val initParams = pack(
      Array.fill(model.p)(0.1),
      Array.fill(model.q)(0.1),
      1.0
    )

    val cssObjective = new DiffFunction[DenseVector[Double]] {
      def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val f = negCssLogLik(params, obs)
        val g = DenseVector.zeros[Double](params.length)
        val h = 1e-5
        for (i <- 0 until params.length) {
          val pp = params.copy; pp(i) += h
          val pm = params.copy; pm(i) -= h
          g(i) = (negCssLogLik(pp, obs) - negCssLogLik(pm, obs)) / (2 * h)
        }
        (f, g)
      }
    }

    val cssParams = new LBFGS[DenseVector[Double]](maxIter = 100, m = 5)
      .minimize(cssObjective, initParams)

    if (cssOnly) {
      val (phi, theta, sigma2) = unpack(cssParams)
      val kalmanModel = model.toKalmanModel(phi, theta, sigma2)
      val initial     = model.initialState(phi, theta, sigma2)
      val ll          = LogLikelihoodAccumulator.cssLogLikelihood(initial, kalmanModel, obs)
      return ArimaFit(
        phi, theta, sigma2, ll, diffed.length,
        Array.fill(model.p + model.q + 1)(Double.NaN),
        cssOnly = true
      )
    }

    val mleObjective = new DiffFunction[DenseVector[Double]] {
      def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val f = negLogLik(params, obs)
        val g = DenseVector.zeros[Double](params.length)
        val h = 1e-5
        for (i <- 0 until params.length) {
          val pp = params.copy; pp(i) += h
          val pm = params.copy; pm(i) -= h
          g(i) = (negLogLik(pp, obs) - negLogLik(pm, obs)) / (2 * h)
        }
        (f, g)
      }
    }

    val mleParams = new LBFGS[DenseVector[Double]](maxIter = 200, m = 5)
      .minimize(mleObjective, cssParams)

    val (phi, theta, sigma2) = unpack(mleParams)
    val kalmanModel = model.toKalmanModel(phi, theta, sigma2)
    val initial     = model.initialState(phi, theta, sigma2)
    val ll          = LogLikelihoodAccumulator.logLikelihood(initial, kalmanModel, obs)
    val ses = if (cssOnly) Array.fill(model.p + model.q + 1)(Double.NaN)
    else standardErrors(mleParams, obs)

    ArimaFit(phi, theta, sigma2, ll, diffed.length, ses, cssOnly)
  }

  private def difference(y: Array[Double], d: Int): Array[Double] =
    if (d == 0) y
    else difference(y.zip(y.tail).map { case (a, b) => b - a }, d - 1)
}

case class ArimaFit(
                     phi:     Array[Double],
                     theta:   Array[Double],
                     sigma2:  Double,
                     logLik:  Double,
                     n:       Int,
                     se:      Array[Double],
                     cssOnly: Boolean
                   ) {
  private val k = phi.length + theta.length + 1

  def aic:  Double = -2 * logLik + 2 * k
  def aicc: Double = aic + (2.0 * k * (k + 1)) / (n - k - 1)
  def bic:  Double = -2 * logLik + math.log(n) * k

  def sigma2SE: Double = sigma2 * se(phi.length + theta.length)

  override def toString: String = {
    val params = phi ++ theta ++ Array(sigma2)
    val ses    = se.take(phi.length + theta.length) ++ Array(sigma2SE)
    val names  = phi.indices.map(i => s"phi${i+1}") ++
      theta.indices.map(i => s"theta${i+1}") ++
      Seq("sigma2")

    val rows = names.zip(params).zip(ses).map { case ((name, est), se) =>
      f"  $name%-10s  $est%10.4f  $se%10.4f"
    }.mkString("\n")

    s"""ArimaFit (${if (cssOnly) "CSS" else "MLE"}):
       |${"  " + f"${""}%-10s  ${"estimate"}%10s  ${"std.err"}%10s"}
       |$rows
       |  logLik   = ${f"$logLik%.4f"}
       |  AIC      = ${f"$aic%.4f"}
       |  AICc     = ${f"${aicc}%.4f"}
       |  BIC      = ${f"$bic%.4f"}""".stripMargin
  }
}