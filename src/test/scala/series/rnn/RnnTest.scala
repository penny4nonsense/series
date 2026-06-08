package series.rnn

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class RnnTest extends AnyFunSuite {

  val rng = new scala.util.Random(42)

  def arSeries(n: Int, k: Int = 1, phi: Double = 0.8): DenseMatrix[Double] = {
    val Y = DenseMatrix.zeros[Double](n, k)
    for (j <- 0 until k)
      for (t <- 1 until n)
        Y(t, j) = phi * Y(t-1, j) + rng.nextGaussian() * 0.3
    Y
  }

  def trendSeries(n: Int, k: Int = 2): DenseMatrix[Double] = {
    DenseMatrix.tabulate(n, k) { (t, j) =>
      0.05 * t * (j + 1) + rng.nextGaussian() * 0.2
    }
  }

  // ── RnnConfig ──────────────────────────────────────────────────────────────

  test("small config has correct defaults") {
    val c = RnnConfig.small
    assert(c.cellType == LSTM)
    assert(c.hiddenDim == 32)
    assert(c.nLayers == 1)
  }

  test("medium config has larger hiddenDim") {
    assert(RnnConfig.medium.hiddenDim == 64)
    assert(RnnConfig.medium.nLayers == 2)
  }

  test("effectiveWindow uses period when period > 1") {
    val c = RnnConfig(period = 12, horizon = 1)
    assert(c.effectiveWindow(200) == 24)
  }

  test("effectiveWindow caps at n/3") {
    val c = RnnConfig(period = 1, horizon = 1)
    assert(c.effectiveWindow(30) == 10)
  }

  test("effectiveWindow respects explicit window") {
    val c = RnnConfig(window = 20)
    assert(c.effectiveWindow(100) == 20)
  }

  // ── LossType ───────────────────────────────────────────────────────────────

  test("MSE loss is zero for perfect prediction") {
    val pred   = DenseVector(1.0, 2.0, 3.0)
    val target = DenseVector(1.0, 2.0, 3.0)
    val (l, g) = LossType.loss(pred, target, MSE)
    assert(math.abs(l) < 1e-10)
    assert(norm(g) < 1e-10)
  }

  test("MSE gradient points in direction of error") {
    val pred   = DenseVector(2.0)
    val target = DenseVector(1.0)
    val (_, g) = LossType.loss(pred, target, MSE)
    assert(g(0) > 0)
  }

  test("MAE loss is correct") {
    val pred   = DenseVector(3.0)
    val target = DenseVector(1.0)
    val (l, _) = LossType.loss(pred, target, MAE)
    assert(math.abs(l - 2.0) < 1e-10)
  }

  test("Huber loss equals MSE for small errors") {
    val pred   = DenseVector(1.1)
    val target = DenseVector(1.0)
    val (lH, _) = LossType.loss(pred, target, Huber)
    val (lM, _) = LossType.loss(pred, target, MSE)
    assert(math.abs(lH - lM / 2.0) < 1e-10)  // Huber = 0.5*d^2 for |d|<=1
  }

  // ── RnnNormalizer ──────────────────────────────────────────────────────────

  test("normalizer transforms to zero mean") {
    val Y = trendSeries(100)
    val n = new RnnNormalizer(2)
    n.fit(Y)
    val Yn = n.transform(Y)
    for (j <- 0 until 2) {
      val mean = Yn(::, j).toArray.sum / 100
      assert(math.abs(mean) < 0.01)
    }
  }

  test("normalizer inverse recovers original") {
    val Y  = trendSeries(100)
    val n  = new RnnNormalizer(2)
    n.fit(Y)
    val Yn = n.transform(Y)
    val Yr = n.inverseTransform(Yn)
    for (i <- 0 until 100; j <- 0 until 2)
      assert(math.abs(Y(i,j) - Yr(i,j)) < 1e-8)
  }

  // ── Cells ─────────────────────────────────────────────────────────────────

  test("VanillaCell output has correct dimension") {
    val r   = new scala.util.Random(42)
    val w   = VanillaWeights(GateWeights.random(16, 4, r))
    val x   = DenseVector.fill(4)(r.nextGaussian())
    val s   = VanillaState(DenseVector.zeros[Double](16))
    val (ns, _) = VanillaCell.forward(x, s, w, None)
    assert(ns.h.length == 16)
  }

  test("LstmCell output and cell state have correct dimensions") {
    val r   = new scala.util.Random(42)
    val w   = LstmWeights.random(16, 4, r)
    val x   = DenseVector.fill(4)(r.nextGaussian())
    val s   = LstmState(DenseVector.zeros[Double](16), DenseVector.zeros[Double](16))
    val (ns, _) = LstmCell.forward(x, s, w, None)
    assert(ns.h.length == 16)
    assert(ns.c.length == 16)
  }

  test("GruCell output has correct dimension") {
    val r   = new scala.util.Random(42)
    val w   = GruWeights.random(16, 4, r)
    val x   = DenseVector.fill(4)(r.nextGaussian())
    val s   = GruState(DenseVector.zeros[Double](16))
    val (ns, _) = GruCell.forward(x, s, w, None)
    assert(ns.h.length == 16)
  }

  test("LSTM gate values are in (0,1)") {
    val r   = new scala.util.Random(42)
    val w   = LstmWeights.random(8, 3, r)
    val x   = DenseVector.fill(3)(r.nextGaussian())
    val s   = LstmState(DenseVector.zeros[Double](8), DenseVector.zeros[Double](8))
    val (_, cache) = LstmCell.forward(x, s, w, None)
    val c = cache.asInstanceOf[LstmCache]
    assert(c.f.toArray.forall(v => v > 0 && v < 1))
    assert(c.i.toArray.forall(v => v > 0 && v < 1))
    assert(c.o.toArray.forall(v => v > 0 && v < 1))
  }

  // ── Vanilla RNN fit ────────────────────────────────────────────────────────

  test("Vanilla RNN fit returns RnnFit") {
    val Y   = arSeries(100, k=1)
    val cfg = RnnConfig(cellType=Vanilla, hiddenDim=8, nLayers=1, epochs=5)
    val fit = RnnModel(cfg).fit(Y)
    assert(fit.nSeries == 1)
  }

  test("LSTM fit returns non-NaN loss") {
    val Y   = arSeries(100, k=2)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=8, nLayers=1, epochs=5)
    val fit = RnnModel(cfg).fit(Y)
    assert(!fit.history.trainLosses.last.isNaN)
  }

  test("GRU fit returns non-NaN loss") {
    val Y   = arSeries(100, k=2)
    val cfg = RnnConfig(cellType=GRU, hiddenDim=8, nLayers=1, epochs=5)
    val fit = RnnModel(cfg).fit(Y)
    assert(!fit.history.trainLosses.last.isNaN)
  }

  test("multilayer LSTM fit works") {
    val Y   = arSeries(150, k=2)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=16, nLayers=2, epochs=5)
    val fit = RnnModel(cfg).fit(Y)
    assert(!fit.history.trainLosses.last.isNaN)
  }

  test("training loss decreases over epochs") {
    val Y   = trendSeries(200, k=1)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=16, nLayers=1, epochs=20,
                        valFraction=0.1, patience=20)
    val fit = RnnModel(cfg).fit(Y)
    val losses = fit.history.trainLosses
    // First loss should exceed last (training improves)
    assert(losses.head > losses.last)
  }

  // ── Forecast ──────────────────────────────────────────────────────────────

  test("forecast has correct dimensions") {
    val Y   = arSeries(150, k=2)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=8, nLayers=1, epochs=5, horizon=5)
    val fit = RnnModel(cfg).fit(Y)
    val fc  = fit.forecast(Y, h=5)
    assert(fc.rows == 5)
    assert(fc.cols == 2)
  }

  test("forecast values are finite") {
    val Y   = arSeries(150, k=2)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=8, nLayers=1, epochs=5)
    val fit = RnnModel(cfg).fit(Y)
    val fc  = fit.forecast(Y)
    assert(fc.toArray.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("MC forecast has correct dimensions") {
    val Y   = arSeries(150, k=2)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=8, nLayers=1, epochs=5,
                        horizon=3, mcSamples=20)
    val fit = RnnModel(cfg).fit(Y)
    val fc  = fit.forecastWithIntervals(Y, h=3, nSamples=20)
    assert(fc.mean.rows == 3 && fc.mean.cols == 2)
    assert(fc.lower.rows == 3)
    assert(fc.upper.rows == 3)
  }

  test("prediction intervals are ordered") {
    val Y   = arSeries(200, k=1)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=16, nLayers=1, epochs=10,
                        horizon=5, mcSamples=50, dropout=0.2)
    val fit = RnnModel(cfg).fit(Y)
    val fc  = fit.forecastWithIntervals(Y, h=5, nSamples=50)
    for (i <- 0 until 5; j <- 0 until 1) {
      assert(fc.lower(i,j) <= fc.mean(i,j) + 1e-8)
      assert(fc.mean(i,j)  <= fc.upper(i,j) + 1e-8)
    }
  }

  test("fitted values have correct dimensions") {
    val Y   = arSeries(150, k=2)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=8, nLayers=1, epochs=5)
    val fit = RnnModel(cfg).fit(Y)
    val fv  = fit.fitted(Y)
    assert(fv.cols == 2)
    assert(fv.rows > 0)
  }

  test("toString is readable") {
    val Y   = arSeries(100, k=1)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=8, epochs=5)
    val fit = RnnModel(cfg).fit(Y)
    assert(fit.toString.contains("LSTM"))
    assert(fit.toString.contains("Window"))
  }

  // ── Different loss functions ───────────────────────────────────────────────

  test("MAE loss training works") {
    val Y   = arSeries(100, k=1)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=8, epochs=5, lossType=MAE)
    val fit = RnnModel(cfg).fit(Y)
    assert(!fit.history.trainLosses.last.isNaN)
  }

  test("Huber loss training works") {
    val Y   = arSeries(100, k=1)
    val cfg = RnnConfig(cellType=LSTM, hiddenDim=8, epochs=5, lossType=Huber)
    val fit = RnnModel(cfg).fit(Y)
    assert(!fit.history.trainLosses.last.isNaN)
  }
}
