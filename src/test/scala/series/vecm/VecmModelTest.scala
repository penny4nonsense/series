package series.vecm

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class VecmModelTest extends AnyFunSuite {

  // Simulate cointegrated bivariate system
  // y1_t = y1_{t-1} + e1_t
  // y2_t = y1_t + u_t  (cointegrated with y1)
  def simulateCointegrated(
                            n:    Int,
                            seed: Long = 42
                          ): DenseMatrix[Double] = {
    val rng = new scala.util.Random(seed)
    val Y   = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      Y(t, 0) = Y(t-1, 0) + rng.nextGaussian() * 0.1
      Y(t, 1) = Y(t, 0) + rng.nextGaussian() * 0.1
    }
    Y
  }

  test("requires at least 2 series") {
    assertThrows[IllegalArgumentException] {
      VecmModel(k=1, p=2, r=1)
    }
  }

  test("requires positive lag order") {
    assertThrows[IllegalArgumentException] {
      VecmModel(k=2, p=0, r=1)
    }
  }

  test("rank must be less than k") {
    assertThrows[IllegalArgumentException] {
      VecmModel(k=2, p=2, r=2)
    }
  }

  test("rank must be positive") {
    assertThrows[IllegalArgumentException] {
      VecmModel(k=2, p=2, r=0)
    }
  }

  test("alpha has correct dimensions") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    assert(fit.alpha.rows == 2 && fit.alpha.cols == 1)
  }

  test("beta has correct dimensions") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    assert(fit.beta.rows == 2 && fit.beta.cols == 1)
  }

  test("Pi = alpha * beta'") {
    val Y    = simulateCointegrated(200)
    val fit  = VecmModel(k=2, p=2, r=1).fit(Y)
    val diff = fit.Pi - fit.alpha * fit.beta.t
    assert(norm(diff.toDenseVector) < 1e-8)
  }

  test("Sigma is positive definite") {
    val Y    = simulateCointegrated(200)
    val fit  = VecmModel(k=2, p=2, r=1).fit(Y)
    val eigs = eigSym(fit.Sigma)
    assert(eigs.eigenvalues.toArray.forall(_ > 0.0))
  }

  test("log likelihood is finite") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    assert(!fit.logLik.isNaN && !fit.logLik.isInfinite)
  }

  test("AIC BIC HQIC are finite") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    assert(!fit.aic.isNaN && !fit.bic.isNaN && !fit.hqic.isNaN)
  }

  test("cointegrating relationship has correct length") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    assert(fit.cointegratingRelationship(0).length == Y.rows)
  }

  test("cointegrating relationship is approximately stationary") {
    // beta' Y should be stationary for cointegrated system
    val Y    = simulateCointegrated(500)
    val fit  = VecmModel(k=2, p=2, r=1).fit(Y)
    val z    = fit.cointegratingRelationship(0)
    val mean = z.sum / z.length
    val v    = z.map(x => math.pow(x - mean, 2)).sum / z.length
    // Variance should be bounded — much smaller than original series
    assert(v < 5.0)
  }

  test("toVarMatrices returns p matrices") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    assert(fit.toVarMatrices.length == 2)
  }

  test("companion matrix has correct dimensions") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    val C   = fit.companionMatrix
    assert(C.rows == 4 && C.cols == 4)
  }

  test("eigenvalues have correct count") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    assert(fit.eigenvalues.length == 1)
  }

  test("eigenvalues are positive") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    assert(fit.eigenvalues.forall(_ > 0.0))
  }

  test("toString contains key fields") {
    val Y   = simulateCointegrated(200)
    val fit = VecmModel(k=2, p=2, r=1).fit(Y)
    val str = fit.toString
    assert(str.contains("beta"))
    assert(str.contains("alpha"))
    assert(str.contains("AIC"))
  }
}