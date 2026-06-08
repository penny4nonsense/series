package series.rnn

import breeze.linalg._

/**
 * Stacked RNN: nLayers of RnnLayer.
 *
 * Layer 0 takes the input sequence.
 * Layer l takes the output sequence of layer l-1.
 * Final layer output goes to the output projection.
 *
 * Output projection: W_hy (k x hiddenDim), b_y (k)
 * Prediction: y_hat = W_hy * h_T + b_y  (last hidden state of final layer)
 *
 * Gradient computation (computeGradients) is separated from the optimizer
 * step (applyGradients) so the raw analytical gradients can be inspected,
 * e.g. for finite-difference gradient checking.
 *
 * @param config   RNN configuration
 * @param inputDim number of input series (k)
 * @param rng      random number generator
 */
class StackedRnn(config: RnnConfig, val inputDim: Int, rng: scala.util.Random) {

  val layers: Array[RnnLayer] = Array.tabulate(config.nLayers) { l =>
    val inDim = if (l == 0) inputDim else config.hiddenDim
    new RnnLayer(config.cellType, config.hiddenDim, inDim, rng)
  }

  // Output projection: hiddenDim -> inputDim
  var Why: DenseMatrix[Double] = DenseMatrix.tabulate(inputDim, config.hiddenDim) { (_, _) =>
    rng.nextGaussian() * math.sqrt(2.0 / config.hiddenDim)
  }
  var by: DenseVector[Double] = DenseVector.zeros[Double](inputDim)

  // Adam states for output projection
  var adamWhy: AdamState    = AdamState.zeros(inputDim, config.hiddenDim)
  var adamBy:  AdamStateVec = AdamStateVec.zeros(inputDim)

  // Adam states for each layer's weights (stored as maps)
  val adamStates: Array[scala.collection.mutable.Map[String, Any]] =
    Array.fill(config.nLayers)(scala.collection.mutable.Map.empty)

  for (l <- 0 until config.nLayers) {
    val inDim = if (l == 0) inputDim else config.hiddenDim
    val h     = config.hiddenDim
    config.cellType match {
      case Vanilla =>
        adamStates(l)("wx") = AdamState.zeros(h, inDim)
        adamStates(l)("wh") = AdamState.zeros(h, h)
        adamStates(l)("b")  = AdamStateVec.zeros(h)
      case LSTM =>
        for (g <- Seq("f","i","g","o")) {
          adamStates(l)(s"wx_$g") = AdamState.zeros(h, inDim)
          adamStates(l)(s"wh_$g") = AdamState.zeros(h, h)
          adamStates(l)(s"b_$g")  = AdamStateVec.zeros(h)
        }
      case GRU =>
        for (g <- Seq("r","z","n")) {
          adamStates(l)(s"wx_$g") = AdamState.zeros(h, inDim)
          adamStates(l)(s"wh_$g") = AdamState.zeros(h, h)
          adamStates(l)(s"b_$g")  = AdamStateVec.zeros(h)
        }
    }
  }

  /**
   * Forward pass through all layers.
   *
   * @param window    input window (T x inputDim) — normalized
   * @param dropout   dropout rate
   * @param training  true during training
   * @return (prediction: inputDim, layerCaches, finalStates)
   */
  def forward(
    window:   DenseMatrix[Double],
    dropout:  Double,
    training: Boolean
  ): (DenseVector[Double], Array[Any], Array[CellState]) = {
    val T = window.rows
    var xs: Array[DenseVector[Double]] = Array.tabulate(T)(t => window(t, ::).t.copy)

    val caches      = new Array[Any](config.nLayers)
    val finalStates = new Array[CellState](config.nLayers)

    for (l <- 0 until config.nLayers) {
      val initState = CellState.zeros(config.cellType, config.hiddenDim)
      val (outputs, finalState, layerCaches) =
        layers(l).forward(xs, initState, dropout, training)
      caches(l)      = layerCaches
      finalStates(l) = finalState
      xs             = outputs
    }

    val hLast = xs.last
    val pred  = Why * hLast + by
    (pred, caches, finalStates)
  }

  /**
   * Computes raw analytical gradients for all parameters.
   * Does NOT apply any optimizer step or gradient clipping.
   *
   * @param dPred   gradient of loss w.r.t. prediction
   * @param caches  forward pass caches
   * @param window  input window
   * @return StackedGrads with output projection + per-layer gradients
   */
  def computeGradients(
    dPred:   DenseVector[Double],
    caches:  Array[Any],
    window:  DenseMatrix[Double]
  ): StackedGrads = {
    val T = window.rows

    // Recompute layer outputs (no dropout) to get hLast for output projection
    val layerOutputs = new Array[Array[DenseVector[Double]]](config.nLayers)
    var xs: Array[DenseVector[Double]] = Array.tabulate(T)(t => window(t, ::).t.copy)
    for (l <- 0 until config.nLayers) {
      val initState = CellState.zeros(config.cellType, config.hiddenDim)
      val (outputs, _, _) = layers(l).forward(xs, initState, 0.0, training = false)
      layerOutputs(l) = outputs
      xs = outputs
    }

    val hLast  = layerOutputs(config.nLayers - 1).last
    val dWhy   = dPred.toDenseMatrix.t * hLast.toDenseMatrix
    val dBy    = dPred
    val dhLast = Why.t * dPred

    var dOutputs: Array[DenseVector[Double]] = Array.tabulate(T) { t =>
      if (t == T - 1) dhLast else DenseVector.zeros[Double](config.hiddenDim)
    }

    val layerGrads = new Array[Any](config.nLayers)
    for (l <- (0 until config.nLayers).reverse) {
      val (dInputs, grads) = layers(l).backward(dOutputs, caches(l), 0.0)
      layerGrads(l) = grads
      dOutputs = dInputs
    }

    StackedGrads(dWhy = dWhy, dBy = dBy, layerGrads = layerGrads)
  }

  /**
   * Applies gradient clipping + Adam updates given computed gradients.
   * Mutates all weights in place.
   */
  def applyGradients(grads: StackedGrads): Unit = {
    val lr = config.learningRate

    // Output projection
    val clipScale = math.min(1.0, config.clipValue /
      math.max(math.sqrt(breeze.linalg.sum(grads.dWhy *:* grads.dWhy) +
        (grads.dBy dot grads.dBy)), 1e-10))
    val (newWhy, newAdamWhy) = Adam.updateMatrix(Why, grads.dWhy * clipScale, adamWhy, lr)
    val (newBy,  newAdamBy)  = Adam.updateVector(by, grads.dBy * clipScale, adamBy, lr)
    Why = newWhy; adamWhy = newAdamWhy
    by  = newBy;  adamBy  = newAdamBy

    for (l <- 0 until config.nLayers)
      applyLayerGrads(l, grads.layerGrads(l))
  }

  /** Convenience: compute + apply in one call (training loop uses this) */
  def backward(
    dPred:   DenseVector[Double],
    caches:  Array[Any],
    window:  DenseMatrix[Double],
    dropout: Double
  ): Unit = applyGradients(computeGradients(dPred, caches, window))

  private def applyLayerGrads(l: Int, grads: Any): Unit = {
    val layer = layers(l)
    val lr    = config.learningRate
    val clip  = config.clipValue

    config.cellType match {
      case Vanilla =>
        val g  = grads.asInstanceOf[VanillaGrads]
        val w  = layer.weights.asInstanceOf[VanillaWeights]
        val cs = adamStates(l)
        val scale = GradientClipper.clipNorm(Seq(g.gate.dWx, g.gate.dWh), clip)
        val (nWx, nSWx) = Adam.updateMatrix(w.gate.Wx, g.gate.dWx * scale,
          cs("wx").asInstanceOf[AdamState], lr)
        val (nWh, nSWh) = Adam.updateMatrix(w.gate.Wh, g.gate.dWh * scale,
          cs("wh").asInstanceOf[AdamState], lr)
        val (nb,  nSb)  = Adam.updateVector(w.gate.b, g.gate.db * scale,
          cs("b").asInstanceOf[AdamStateVec], lr)
        w.gate.Wx := nWx; cs("wx") = nSWx
        w.gate.Wh := nWh; cs("wh") = nSWh
        w.gate.b  := nb;  cs("b")  = nSb

      case LSTM =>
        val g  = grads.asInstanceOf[LstmGrads]
        val w  = layer.weights.asInstanceOf[LstmWeights]
        val cs = adamStates(l)
        val allGrads = Seq(g.wf.dWx,g.wf.dWh,g.wi.dWx,g.wi.dWh,
                          g.wg.dWx,g.wg.dWh,g.wo.dWx,g.wo.dWh)
        val scale = GradientClipper.clipNorm(allGrads, clip)
        def applyGate(gw: GateWeights, gg: GateGrads, key: String): Unit = {
          val (nWx, nSWx) = Adam.updateMatrix(gw.Wx, gg.dWx*scale, cs(s"wx_$key").asInstanceOf[AdamState], lr)
          val (nWh, nSWh) = Adam.updateMatrix(gw.Wh, gg.dWh*scale, cs(s"wh_$key").asInstanceOf[AdamState], lr)
          val (nb,  nSb)  = Adam.updateVector(gw.b,  gg.db*scale,  cs(s"b_$key").asInstanceOf[AdamStateVec], lr)
          gw.Wx := nWx; cs(s"wx_$key") = nSWx
          gw.Wh := nWh; cs(s"wh_$key") = nSWh
          gw.b  := nb;  cs(s"b_$key")  = nSb
        }
        applyGate(w.wf, g.wf, "f")
        applyGate(w.wi, g.wi, "i")
        applyGate(w.wg, g.wg, "g")
        applyGate(w.wo, g.wo, "o")

      case GRU =>
        val g  = grads.asInstanceOf[GruGrads]
        val w  = layer.weights.asInstanceOf[GruWeights]
        val cs = adamStates(l)
        val allGrads = Seq(g.wr.dWx,g.wr.dWh,g.wz.dWx,g.wz.dWh,g.wn.dWx,g.wn.dWh)
        val scale = GradientClipper.clipNorm(allGrads, clip)
        def applyGate(gw: GateWeights, gg: GateGrads, key: String): Unit = {
          val (nWx, nSWx) = Adam.updateMatrix(gw.Wx, gg.dWx*scale, cs(s"wx_$key").asInstanceOf[AdamState], lr)
          val (nWh, nSWh) = Adam.updateMatrix(gw.Wh, gg.dWh*scale, cs(s"wh_$key").asInstanceOf[AdamState], lr)
          val (nb,  nSb)  = Adam.updateVector(gw.b,  gg.db*scale,  cs(s"b_$key").asInstanceOf[AdamStateVec], lr)
          gw.Wx := nWx; cs(s"wx_$key") = nSWx
          gw.Wh := nWh; cs(s"wh_$key") = nSWh
          gw.b  := nb;  cs(s"b_$key")  = nSb
        }
        applyGate(w.wr, g.wr, "r")
        applyGate(w.wz, g.wz, "z")
        applyGate(w.wn, g.wn, "n")
    }
  }
}

/**
 * Raw analytical gradients for the full stacked RNN.
 *
 * @param dWhy       output projection weight gradient
 * @param dBy        output projection bias gradient
 * @param layerGrads per-layer cell gradients (VanillaGrads/LstmGrads/GruGrads)
 */
case class StackedGrads(
  dWhy:       DenseMatrix[Double],
  dBy:        DenseVector[Double],
  layerGrads: Array[Any]
)
