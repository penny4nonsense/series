package series.rnn

import breeze.linalg._

/**
 * Finite-difference gradient checking for the stacked RNN.
 *
 * Verifies analytical BPTT gradients against numerical central-difference
 * estimates. This is the test that proves the backward passes of the
 * Vanilla / LSTM / GRU cells are mathematically correct — not merely that
 * training loss happens to decrease.
 *
 * Method:
 *   For each parameter w:
 *     numeric  = (L(w + eps) - L(w - eps)) / (2 * eps)
 *     analytic = computeGradients(...)
 *     relError = |analytic - numeric| / max(|analytic|, |numeric|, eps)
 *
 * All checks run with dropout = 0 so the loss is deterministic.
 *
 * @param eps   perturbation size (central differences, O(eps^2) error)
 */
class GradientCheck(eps: Double = 1e-5) {

  /**
   * Runs a gradient check on a single (window, target) example.
   *
   * @param net    the stacked RNN (weights are perturbed and restored)
   * @param window input window (T x k)
   * @param target target vector (k)
   * @param loss   loss type
   * @return GradientCheckResult with max and mean relative error
   */
  def check(
    net:    StackedRnn,
    window: DenseMatrix[Double],
    target: DenseVector[Double],
    loss:   LossType
  ): GradientCheckResult = {
    // Analytical gradients
    val (pred, caches, _) = net.forward(window, dropout = 0.0, training = false)
    val (_, dLoss)        = LossType.loss(pred, target, loss)
    val grads             = net.computeGradients(dLoss, caches, window)

    val relErrors = scala.collection.mutable.ArrayBuffer[Double]()

    // Loss as a pure function of current weights
    def lossAt(): Double = {
      val (p, _, _) = net.forward(window, dropout = 0.0, training = false)
      LossType.loss(p, target, loss)._1
    }

    // ── Check output projection Why ──────────────────────────────────────────
    checkMatrix(net.Why, grads.dWhy, lossAt, relErrors)
    checkVector(net.by,  grads.dBy,  lossAt, relErrors)

    // ── Check each layer's cell weights ────────────────────────────────────────
    for (l <- 0 until net.layers.length) {
      val layer = net.layers(l)
      net.layers(l).cellType match {
        case Vanilla =>
          val w = layer.weights.asInstanceOf[VanillaWeights]
          val g = grads.layerGrads(l).asInstanceOf[VanillaGrads]
          checkGate(w.gate, g.gate, lossAt, relErrors)

        case LSTM =>
          val w = layer.weights.asInstanceOf[LstmWeights]
          val g = grads.layerGrads(l).asInstanceOf[LstmGrads]
          checkGate(w.wf, g.wf, lossAt, relErrors)
          checkGate(w.wi, g.wi, lossAt, relErrors)
          checkGate(w.wg, g.wg, lossAt, relErrors)
          checkGate(w.wo, g.wo, lossAt, relErrors)

        case GRU =>
          val w = layer.weights.asInstanceOf[GruWeights]
          val g = grads.layerGrads(l).asInstanceOf[GruGrads]
          checkGate(w.wr, g.wr, lossAt, relErrors)
          checkGate(w.wz, g.wz, lossAt, relErrors)
          checkGate(w.wn, g.wn, lossAt, relErrors)
      }
    }

    GradientCheckResult(
      maxRelError  = if (relErrors.isEmpty) 0.0 else relErrors.max,
      meanRelError = if (relErrors.isEmpty) 0.0 else relErrors.sum / relErrors.length,
      nChecked     = relErrors.length
    )
  }

  private def checkGate(
    w:         GateWeights,
    g:         GateGrads,
    lossAt:    () => Double,
    relErrors: scala.collection.mutable.ArrayBuffer[Double]
  ): Unit = {
    checkMatrix(w.Wx, g.dWx, lossAt, relErrors)
    checkMatrix(w.Wh, g.dWh, lossAt, relErrors)
    checkVector(w.b,  g.db,  lossAt, relErrors)
  }

  private def checkMatrix(
    param:     DenseMatrix[Double],
    analytic:  DenseMatrix[Double],
    lossAt:    () => Double,
    relErrors: scala.collection.mutable.ArrayBuffer[Double]
  ): Unit = {
    // Sample a subset of entries for large matrices (keep the check fast)
    val entries = sampleEntries(param.rows, param.cols)
    for ((i, j) <- entries) {
      val orig = param(i, j)
      param(i, j) = orig + eps
      val lPlus  = lossAt()
      param(i, j) = orig - eps
      val lMinus = lossAt()
      param(i, j) = orig  // restore

      val numeric = (lPlus - lMinus) / (2 * eps)
      relErrors += relError(analytic(i, j), numeric)
    }
  }

  private def checkVector(
    param:     DenseVector[Double],
    analytic:  DenseVector[Double],
    lossAt:    () => Double,
    relErrors: scala.collection.mutable.ArrayBuffer[Double]
  ): Unit = {
    val idxs = sampleIndices(param.length)
    for (i <- idxs) {
      val orig = param(i)
      param(i) = orig + eps
      val lPlus  = lossAt()
      param(i) = orig - eps
      val lMinus = lossAt()
      param(i) = orig

      val numeric = (lPlus - lMinus) / (2 * eps)
      relErrors += relError(analytic(i), numeric)
    }
  }

  private def relError(analytic: Double, numeric: Double): Double = {
    val denom = math.max(math.max(math.abs(analytic), math.abs(numeric)), eps)
    math.abs(analytic - numeric) / denom
  }

  // Sample up to maxSamples entries deterministically (stride sampling)
  private val maxSamples = 12
  private def sampleEntries(rows: Int, cols: Int): Seq[(Int, Int)] = {
    val all = for (i <- 0 until rows; j <- 0 until cols) yield (i, j)
    stride(all, maxSamples)
  }
  private def sampleIndices(n: Int): Seq[Int] = stride(0 until n, maxSamples)

  private def stride[A](xs: Seq[A], k: Int): Seq[A] = {
    if (xs.length <= k) xs
    else {
      val step = xs.length.toDouble / k
      (0 until k).map(i => xs((i * step).toInt))
    }
  }
}

/**
 * Result of a finite-difference gradient check.
 *
 * @param maxRelError  worst relative error across all checked parameters
 * @param meanRelError mean relative error
 * @param nChecked     number of parameter entries checked
 */
case class GradientCheckResult(
  maxRelError:  Double,
  meanRelError: Double,
  nChecked:     Int
) {
  /** A check passes if the worst relative error is below the tolerance */
  def passes(tol: Double = 1e-4): Boolean = maxRelError < tol

  override def toString: String =
    f"GradientCheck(maxRelError=$maxRelError%.2e, " +
    f"meanRelError=$meanRelError%.2e, nChecked=$nChecked)"
}
