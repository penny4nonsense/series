package series.arima

import breeze.linalg._
import breeze.optimize._
import series.kalman._

/**
 * Estimates SARIMA parameters via CSS then exact MLE.
 *
 * @param model SarimaModel specifying orders and seasonal period
 */
class SarimaEstimator(model: SarimaModel) {

  private val nParams = model.p + model.P + model.q + model.Q + 1

  private def unpack(
                      params: DenseVector[Double]
                    ): (Array[Double], Array[Double], Array[Double], Array[Double], Double) = {
    var idx    = 0
    val phi    = params(idx until idx + model.p).toArray; idx += model.p
    val phiS   = params(idx until idx + model.P).toArray; idx += model.P
    val theta  = params(idx until idx + model.q).toArray; idx += model.q
    val thetaS = params(idx until idx + model.Q).toArray; idx += model.Q
    val sigma2 = math.exp(params(idx))
    (phi, phiS, theta, thetaS, sigma2)
  }

  private def pack(
                    phi:    Array[Double],
                    phiS:   Array[Double],
                    theta:  Array[Double],
                    thetaS: Array[Double],
                    sigma2: Double
                  ): DenseVector[Double] =
    DenseVector(phi ++ phiS ++ theta ++ thetaS ++ Array(math.log(sigma2)))

  private def negLogLik(
                         params: DenseVector[Double],
                         obs:    Seq[DenseVector[Double]]
                       ): Double = {
    try {
      val (phi, phiS, theta, thetaS, sigma2) = unpack(params)
      val km      = model.toKalmanModel(phi, phiS, theta, thetaS, sigma2)
      val initial = model.initialState(phi, phiS, theta, thetaS, sigma2)
      -LogLikelihoodAccumulator.logLikelihood(initial, km, obs)
    } catch {
      case _: Exception => Double.MaxValue
    }
  }

  private def negCssLogLik(
                            params: DenseVector[Double],
                            obs:    Seq[DenseVector[Double]]
                          ): Double = {
    try {
      val (phi, phiS, theta, thetaS, sigma2) = unpack(params)
      val km      = model.toKalmanModel(phi, phiS, theta, thetaS, sigma2)
      val initial = model.initialState(phi, phiS, theta, thetaS, sigma2)
      -LogLikelihoodAccumulator.cssLogLikelihood(initial, km, obs)
    } catch {
      case _: Exception => Double.MaxValue
    }
  }

  private def numericalGradient(
                                 f:      DenseVector[Double] => Double,
                                 params: DenseVector[Double]
                               ): DenseVector[Double] = {
    val g = DenseVector.zeros[Double](params.length)
    val h = 1e-5
    for (i <- 0 until params.length) {
      val pp = params.copy; pp(i) += h
      val pm = params.copy; pm(i) -= h
      g(i) = (f(pp) - f(pm)) / (2 * h)
    }
    g
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

  private def difference(y: Array[Double]): Array[Double] = {
    var yd = y
    for (_ <- 0 until model.d)
      yd = yd.zip(yd.tail).map { case (a, b) => b - a }
    for (_ <- 0 until model.D)
      yd = yd.zipWithIndex
        .filter(_._2 >= model.s)
        .map { case (v, i) => v - yd(i - model.s) }
    yd
  }

  def fit(y: Array[Double], cssOnly: Boolean = false): SarimaFit = {
    val diffed = difference(y)
    val obs: Seq[DenseVector[Double]] = diffed.map(v => DenseVector(v)).toIndexedSeq

    val initParams = pack(
      Array.fill(model.p)(0.1),
      Array.fill(model.P)(0.1),
      Array.fill(model.q)(0.1),
      Array.fill(model.Q)(0.1),
      1.0
    )

    val cssObj = new DiffFunction[DenseVector[Double]] {
      def calculate(params: DenseVector[Double]) = {
        val f = negCssLogLik(params, obs)
        val g = numericalGradient(negCssLogLik(_, obs), params)
        (f, g)
      }
    }

    val cssParams = new LBFGS[DenseVector[Double]](maxIter = 100, m = 5)
      .minimize(cssObj, initParams)

    if (cssOnly) {
      val (phi, phiS, theta, thetaS, sigma2) = unpack(cssParams)
      val km      = model.toKalmanModel(phi, phiS, theta, thetaS, sigma2)
      val initial = model.initialState(phi, phiS, theta, thetaS, sigma2)
      val ll      = LogLikelihoodAccumulator.cssLogLikelihood(initial, km, obs)
      return SarimaFit(phi, phiS, theta, thetaS, sigma2, ll,
        diffed.length, Array.fill(nParams)(Double.NaN), cssOnly = true)
    }

    val mleObj = new DiffFunction[DenseVector[Double]] {
      def calculate(params: DenseVector[Double]) = {
        val f = negLogLik(params, obs)
        val g = numericalGradient(negLogLik(_, obs), params)
        (f, g)
      }
    }

    val mleParams = new LBFGS[DenseVector[Double]](maxIter = 200, m = 5)
      .minimize(mleObj, cssParams)

    val (phi, phiS, theta, thetaS, sigma2) = unpack(mleParams)
    val km      = model.toKalmanModel(phi, phiS, theta, thetaS, sigma2)
    val initial = model.initialState(phi, phiS, theta, thetaS, sigma2)
    val ll      = LogLikelihoodAccumulator.logLikelihood(initial, km, obs)
    val ses     = standardErrors(mleParams, obs)

    SarimaFit(phi, phiS, theta, thetaS, sigma2, ll, diffed.length, ses, cssOnly = false)
  }
}

/**
 * Results from SARIMA estimation.
 * Note: phiS = seasonal AR coefficients, thetaS = seasonal MA coefficients.
 */
case class SarimaFit(
                      phi:     Array[Double],
                      phiS:    Array[Double],
                      theta:   Array[Double],
                      thetaS:  Array[Double],
                      sigma2:  Double,
                      logLik:  Double,
                      n:       Int,
                      se:      Array[Double],
                      cssOnly: Boolean
                    ) {
  private val k = phi.length + phiS.length + theta.length + thetaS.length + 1

  def aic:  Double = -2 * logLik + 2 * k
  def aicc: Double = aic + (2.0 * k * (k + 1)) / (n - k - 1)
  def bic:  Double = -2 * logLik + math.log(n) * k

  def sigma2SE: Double = sigma2 * se.last

  override def toString: String = {
    val names  = phi.indices.map(i => s"phi${i+1}") ++
      phiS.indices.map(i => s"Phi${i+1}") ++
      theta.indices.map(i => s"theta${i+1}") ++
      thetaS.indices.map(i => s"Theta${i+1}") ++
      Seq("sigma2")
    val params = phi ++ phiS ++ theta ++ thetaS ++ Array(sigma2)
    val ses    = se.take(k - 1) ++ Array(sigma2SE)

    val rows = names.zip(params).zip(ses).map { case ((name, est), se) =>
      f"  $name%-10s  $est%10.4f  $se%10.4f"
    }.mkString("\n")

    s"""SarimaFit (${if (cssOnly) "CSS" else "MLE"}):
       |${"  " + f"${""}%-10s  ${"estimate"}%10s  ${"std.err"}%10s"}
       |$rows
       |  logLik   = ${f"$logLik%.4f"}
       |  AIC      = ${f"$aic%.4f"}
       |  AICc     = ${f"${aicc}%.4f"}
       |  BIC      = ${f"$bic%.4f"}""".stripMargin
  }
}