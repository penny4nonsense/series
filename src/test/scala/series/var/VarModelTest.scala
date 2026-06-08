package series.`var`

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class VarModelTest extends AnyFunSuite {

  def simulateVAR1(n: Int, seed: Long = 42): DenseMatrix[Double] = {
    val A   = DenseMatrix((0.5, 0.1), (0.0, 0.4))
    val rng = new scala.util.Random(seed)
    val Y   = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      val yhat = A * Y(t-1, ::).t
      Y(t, 0) = yhat(0) + rng.nextGaussian() * 0.1
      Y(t, 1) = yhat(1) + rng.nextGaussian() * 0.1
    }
    Y
  }

  val A: DenseMatrix[Double] = DenseMatrix((0.5, 0.1), (0.0, 0.4))

  test("requires at least 2 series") {
    assertThrows[IllegalArgumentException] {
      VarModel(1, 1)
    }
  }

  test("requires positive lag order") {
    assertThrows[IllegalArgumentException] {
      VarModel(0, 2)
    }
  }

  test("fit returns correct coefficient dimensions") {
    val Y     = simulateVAR1(200)
    val model = VarModel(p=1, k=2)
    val fit   = model.fit(Y)
    assert(fit.B.rows == model.nCoeffs)
    assert(fit.B.cols == 2)
  }

  test("residual covariance is symmetric") {
    val Y     = simulateVAR1(200)
    val model = VarModel(p=1, k=2)
    val fit   = model.fit(Y)
    val diff  = fit.Sigma - fit.Sigma.t
    assert(norm(diff.toDenseVector) < 1e-10)
  }

  test("residual covariance is positive definite") {
    val Y     = simulateVAR1(200)
    val model = VarModel(p=1, k=2)
    val fit   = model.fit(Y)
    val eigs  = eigSym(fit.Sigma)
    assert(eigs.eigenvalues.toArray.forall(_ > 0.0))
  }

  test("log likelihood is finite") {
    val Y     = simulateVAR1(200)
    val model = VarModel(p=1, k=2)
    val fit   = model.fit(Y)
    assert(!fit.logLik.isNaN && !fit.logLik.isInfinite)
  }

  test("AIC BIC HQIC are finite") {
    val Y     = simulateVAR1(200)
    val model = VarModel(p=1, k=2)
    val fit   = model.fit(Y)
    assert(!fit.aic.isNaN)
    assert(!fit.bic.isNaN)
    assert(!fit.hqic.isNaN)
  }

  test("companion matrix has correct dimensions") {
    val Y     = simulateVAR1(200)
    val model = VarModel(p=2, k=2)
    val fit   = model.fit(Y)
    val c     = fit.companionMatrix
    assert(c.rows == 4)
    assert(c.cols == 4)
  }

  test("stationary VAR is detected as stable") {
    val Y     = simulateVAR1(500)
    val model = VarModel(p=1, k=2)
    val fit   = model.fit(Y)
    assert(fit.isStable)
  }

  test("lag matrix has correct dimensions") {
    val Y     = simulateVAR1(200)
    val model = VarModel(p=2, k=2)
    val fit   = model.fit(Y)
    val a1    = fit.lagMatrix(1)
    assert(a1.rows == 2 && a1.cols == 2)
  }

  test("VAR(1) recovers approximate lag coefficients") {
    val Y     = simulateVAR1(1000)
    val model = VarModel(p=1, k=2)
    val fit   = model.fit(Y)
    val a1    = fit.lagMatrix(1)
    assert(math.abs(a1(0,0) - 0.5) < 0.1)
    assert(math.abs(a1(1,1) - 0.4) < 0.1)
  }

  test("standard errors are positive") {
    val Y   = simulateVAR1(200)
    val fit = VarModel(p=1, k=2).fit(Y)
    assert(fit.se.toArray.forall(_ > 0.0))
  }

  test("toString contains key fields") {
    val Y   = simulateVAR1(200)
    val fit = VarModel(p=1, k=2).fit(Y)
    val str = fit.toString
    assert(str.contains("AIC"))
    assert(str.contains("Stable"))
  }
}
