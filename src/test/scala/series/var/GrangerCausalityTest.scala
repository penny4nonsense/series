package series.`var`

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class GrangerCausalityTest extends AnyFunSuite {

  // Simulate where y1 causes y2 but not vice versa
  def simulateCausal(n: Int, seed: Long = 42): DenseMatrix[Double] = {
    val rng = new scala.util.Random(seed)
    val Y   = DenseMatrix.zeros[Double](n, 2)
    for (t <- 1 until n) {
      Y(t, 0) = 0.5 * Y(t-1, 0) + rng.nextGaussian() * 0.1
      Y(t, 1) = 0.4 * Y(t-1, 1) + 0.3 * Y(t-1, 0) + rng.nextGaussian() * 0.1
    }
    Y
  }

  def fitVar(Y: DenseMatrix[Double]): VarFit =
    VarModel(p=1, k=2).fit(Y)

  test("F statistic is positive") {
    val Y   = simulateCausal(200)
    val fit = fitVar(Y)
    val r   = GrangerCausality.test(fit, cause=0, effect=1)
    assert(r.fStatistic > 0.0)
  }

  test("p-value is in [0,1]") {
    val Y   = simulateCausal(200)
    val fit = fitVar(Y)
    val r   = GrangerCausality.test(fit, cause=0, effect=1)
    assert(r.pValue >= 0.0 && r.pValue <= 1.0)
  }

  test("y1 Granger-causes y2 in causal DGP") {
    val Y   = simulateCausal(500)
    val fit = fitVar(Y)
    val r   = GrangerCausality.test(fit, cause=0, effect=1)
    assert(r.isSignificant(0.05))
  }

  test("y2 does not Granger-cause y1 in causal DGP") {
    val Y   = simulateCausal(500)
    val fit = fitVar(Y)
    val r   = GrangerCausality.test(fit, cause=1, effect=0)
    assert(!r.isSignificant(0.05))
  }

  test("testAll returns k x k matrix") {
    val Y     = simulateCausal(200)
    val fit   = fitVar(Y)
    val all   = GrangerCausality.testAll(fit)
    assert(all.length == 2)
    assert(all(0).length == 2)
  }

  test("diagonal of testAll is None") {
    val Y   = simulateCausal(200)
    val fit = fitVar(Y)
    val all = GrangerCausality.testAll(fit)
    assert(all(0)(0).isEmpty)
    assert(all(1)(1).isEmpty)
  }

  test("throws on same cause and effect") {
    val Y   = simulateCausal(200)
    val fit = fitVar(Y)
    assertThrows[IllegalArgumentException] {
      GrangerCausality.test(fit, cause=0, effect=0)
    }
  }

  test("causalityMatrix produces readable output") {
    val Y   = simulateCausal(200)
    val fit = fitVar(Y)
    val str = GrangerCausality.causalityMatrix(fit)
    assert(str.contains("y1"))
    assert(str.contains("y2"))
  }

  test("toString contains key fields") {
    val Y   = simulateCausal(200)
    val fit = fitVar(Y)
    val r   = GrangerCausality.test(fit, cause=0, effect=1)
    assert(r.toString.contains("F("))
    assert(r.toString.contains("p-value"))
  }
}