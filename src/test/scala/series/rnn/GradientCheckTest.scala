package series.rnn

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

/**
 * Finite-difference gradient checks for all three cell types.
 *
 * These tests verify the analytical BPTT gradients are mathematically
 * correct, the same standard applied to the cleands neural net module
 * (gradients checked against finite differences).
 *
 * Tolerance: central differences with eps=1e-5 should match analytical
 * gradients to better than 1e-4 relative error when activations are in
 * the responsive (non-saturated) region.
 */
class GradientCheckTest extends AnyFunSuite {

  val tol = 1e-4

  /**
   * Build a small net with modest weights so activations don't saturate.
   * Saturated tanh/sigmoid units have near-zero local gradient, which makes
   * the relative-error metric numerically unstable (0/0-ish).
   */
  def smallNet(cellType: CellType, k: Int, hidden: Int, layers: Int): StackedRnn = {
    val cfg = RnnConfig(cellType = cellType, hiddenDim = hidden,
      nLayers = layers, dropout = 0.0, lossType = MSE)
    val rng = new scala.util.Random(7)
    val net = new StackedRnn(cfg, k, rng)
    // Shrink all weights to keep pre-activations small
    scaleWeights(net, 0.3)
    net
  }

  def scaleWeights(net: StackedRnn, factor: Double): Unit = {
    net.Why := net.Why * factor
    for (layer <- net.layers) {
      layer.weights match {
        case w: VanillaWeights =>
          w.gate.Wx := w.gate.Wx * factor; w.gate.Wh := w.gate.Wh * factor
        case w: LstmWeights =>
          for (g <- Seq(w.wf, w.wi, w.wg, w.wo)) {
            g.Wx := g.Wx * factor; g.Wh := g.Wh * factor
          }
        case w: GruWeights =>
          for (g <- Seq(w.wr, w.wz, w.wn)) {
            g.Wx := g.Wx * factor; g.Wh := g.Wh * factor
          }
      }
    }
  }

  val rng = new scala.util.Random(123)
  def window(T: Int, k: Int): DenseMatrix[Double] =
    DenseMatrix.tabulate(T, k)((_, _) => rng.nextGaussian() * 0.5)
  def target(k: Int): DenseVector[Double] =
    DenseVector.tabulate(k)(_ => rng.nextGaussian() * 0.5)

  // ── Vanilla ──────────────────────────────────────────────────────────────

  test("Vanilla RNN gradient check (single layer, univariate)") {
    val net = smallNet(Vanilla, k=1, hidden=6, layers=1)
    val r   = new GradientCheck().check(net, window(5,1), target(1), MSE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  test("Vanilla RNN gradient check (single layer, multivariate)") {
    val net = smallNet(Vanilla, k=3, hidden=6, layers=1)
    val r   = new GradientCheck().check(net, window(5,3), target(3), MSE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  test("Vanilla RNN gradient check (two layers)") {
    val net = smallNet(Vanilla, k=2, hidden=5, layers=2)
    val r   = new GradientCheck().check(net, window(6,2), target(2), MSE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  // ── LSTM ────────────────────────────────────────────────────────────────

  test("LSTM gradient check (single layer, univariate)") {
    val net = smallNet(LSTM, k=1, hidden=6, layers=1)
    val r   = new GradientCheck().check(net, window(5,1), target(1), MSE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  test("LSTM gradient check (single layer, multivariate)") {
    val net = smallNet(LSTM, k=3, hidden=6, layers=1)
    val r   = new GradientCheck().check(net, window(5,3), target(3), MSE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  test("LSTM gradient check (two layers)") {
    val net = smallNet(LSTM, k=2, hidden=5, layers=2)
    val r   = new GradientCheck().check(net, window(6,2), target(2), MSE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  // ── GRU ─────────────────────────────────────────────────────────────────

  test("GRU gradient check (single layer, univariate)") {
    val net = smallNet(GRU, k=1, hidden=6, layers=1)
    val r   = new GradientCheck().check(net, window(5,1), target(1), MSE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  test("GRU gradient check (single layer, multivariate)") {
    val net = smallNet(GRU, k=3, hidden=6, layers=1)
    val r   = new GradientCheck().check(net, window(5,3), target(3), MSE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  test("GRU gradient check (two layers)") {
    val net = smallNet(GRU, k=2, hidden=5, layers=2)
    val r   = new GradientCheck().check(net, window(6,2), target(2), MSE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  // ── Loss functions ─────────────────────────────────────────────────────────

  test("LSTM gradient check with MAE loss") {
    val net = smallNet(LSTM, k=2, hidden=5, layers=1)
    val r   = new GradientCheck().check(net, window(5,2), target(2), MAE)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  test("LSTM gradient check with Huber loss") {
    val net = smallNet(LSTM, k=2, hidden=5, layers=1)
    val r   = new GradientCheck().check(net, window(5,2), target(2), Huber)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  test("GRU gradient check with Huber loss") {
    val net = smallNet(GRU, k=2, hidden=5, layers=1)
    val r   = new GradientCheck().check(net, window(5,2), target(2), Huber)
    assert(r.passes(tol), s"gradient check failed: $r")
  }

  // ── Result reporting ───────────────────────────────────────────────────────

  test("gradient check result is readable and nChecked is positive") {
    val net = smallNet(LSTM, k=1, hidden=4, layers=1)
    val r   = new GradientCheck().check(net, window(4,1), target(1), MSE)
    assert(r.nChecked > 0)
    assert(r.toString.contains("GradientCheck"))
  }
}
