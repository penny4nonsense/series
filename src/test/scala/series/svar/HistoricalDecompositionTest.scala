package series.svar

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite
import series.`var`.VarModel

class HistoricalDecompositionTest extends AnyFunSuite {

  def buildSvarAndResult(n: Int = 500): (SvarModel, SvarResult) = {
    val A = DenseMatrix((0.5, 0.1), (0.0, 0.4))
    val rng = new scala.util.Random(42)
    val Y = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      val yhat = A * Y(t - 1, ::).t
      Y(t, 0) = yhat(0) + rng.nextGaussian() * 0.1
      Y(t, 1) = yhat(1) + rng.nextGaussian() * 0.1
    }
    val fit = VarModel(p = 1, k = 2).fit(Y)
    val svar = new SvarModel(fit)
    val result = ShortRunIdentification.cholesky(svar, h = 20)
    (svar, result)
  }

  test("contributions have correct dimensions") {
    val (_, result) = buildSvarAndResult()
    val hd          = HistoricalDecomposition.decompose(result)
    assert(hd.n == result.shocks.rows)
    assert(hd.k == 2)
  }

  test("contributions are finite") {
    val (_, result) = buildSvarAndResult()
    val hd          = HistoricalDecomposition.decompose(result)
    for (t <- 0 until hd.n; i <- 0 until 2; j <- 0 until 2)
      assert(!hd.contributions(t)(i)(j).isNaN)
  }

  test("variance shares are in [0, 1]") {
    val (_, result) = buildSvarAndResult()
    val hd          = HistoricalDecomposition.decompose(result)
    for (i <- 0 until 2; j <- 0 until 2) {
      val vs = hd.varianceShare(i, j)
      assert(vs >= 0.0 && vs <= 1.0)
    }
  }

  test("variance shares sum to 1 for each series") {
    val (_, result) = buildSvarAndResult()
    val hd          = HistoricalDecomposition.decompose(result)
    for (i <- 0 until 2) {
      val total = (0 until 2).map(j => hd.varianceShare(i, j)).sum
      assert(math.abs(total - 1.0) < 1e-6)
    }
  }

  test("total contribution has correct length") {
    val (_, result) = buildSvarAndResult()
    val hd          = HistoricalDecomposition.decompose(result)
    assert(hd.totalContribution(0, 0).length == hd.n)
  }

  test("toString contains variance shares") {
    val (_, result) = buildSvarAndResult()
    val hd          = HistoricalDecomposition.decompose(result)
    assert(hd.toString.contains("Variance shares"))
  }
}