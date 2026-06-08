package series.svar

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite
import series.`var`.VarModel

class SvarModelTest extends AnyFunSuite {

  def simulateVAR1(n: Int, seed: Long = 42): DenseMatrix[Double] = {
    val A = DenseMatrix((0.5, 0.1), (0.0, 0.4))
    val rng = new scala.util.Random(seed)
    val Y = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      val yhat = A * Y(t - 1, ::).t
      Y(t, 0) = yhat(0) + rng.nextGaussian() * 0.1
      Y(t, 1) = yhat(1) + rng.nextGaussian() * 0.1
    }
    Y
  }

  def buildSvar(n: Int = 500, seed: Long = 42): SvarModel = {
    val A = DenseMatrix((0.5, 0.1), (0.0, 0.4))
    val rng = new scala.util.Random(seed)
    val Y = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      val yhat = A * Y(t - 1, ::).t
      Y(t, 0) = yhat(0) + rng.nextGaussian() * 0.1
      Y(t, 1) = yhat(1) + rng.nextGaussian() * 0.1
    }
    val fit = VarModel(p = 1, k = 2).fit(Y)
    new SvarModel(fit)
  }

  test("Cholesky factor is lower triangular") {
    val svar = buildSvar()
    val P    = svar.choleskyFactor
    assert(math.abs(P(0, 1)) < 1e-10)
  }

  test("Cholesky factor satisfies P*P' = Sigma") {
    val svar = buildSvar()
    val P    = svar.choleskyFactor
    val diff = P * P.t - svar.Sigma
    assert(norm(diff.toDenseVector) < 1e-8)
  }

  test("structural IRFs have correct dimensions") {
    val svar  = buildSvar()
    val B0inv = svar.choleskyFactor
    val sirfs = svar.structuralIRF(B0inv, 10)
    assert(sirfs.length == 11)
    assert(sirfs(0).rows == 2 && sirfs(0).cols == 2)
  }

  test("long run multiplier is finite") {
    val svar = buildSvar()
    val C1   = svar.longRunMultiplier
    assert(C1.toArray.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("structural shocks have correct dimensions") {
    val svar   = buildSvar()
    val B0inv  = svar.choleskyFactor
    val shocks = svar.structuralShocks(B0inv)
    assert(shocks.cols == 2)
  }

  test("verifySigma passes for Cholesky impact matrix") {
    val svar  = buildSvar()
    val B0inv = svar.choleskyFactor
    assert(svar.verifySigma(B0inv))
  }

  test("cumulative IRF has correct dimensions") {
    val svar  = buildSvar()
    val B0inv = svar.choleskyFactor
    val cum   = svar.cumulativeIRF(B0inv, 20)
    assert(cum.rows == 2 && cum.cols == 2)
  }
}