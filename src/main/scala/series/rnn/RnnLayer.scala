package series.rnn

import breeze.linalg._

/**
 * A single RNN layer: unrolls a cell over a sequence and computes BPTT gradients.
 *
 * Forward pass: for t=0..T-1, compute (h_t, cache_t) = cell.forward(x_t, h_{t-1})
 * Backward pass: for t=T-1..0, accumulate gradients via BPTT
 *
 * @param cellType  which cell to use
 * @param hiddenDim hidden state dimension
 * @param inputDim  input dimension
 * @param rng       random number generator
 */
class RnnLayer(
  val cellType:  CellType,
  val hiddenDim: Int,
  val inputDim:  Int,
  rng:           scala.util.Random
) {
  // ── Weights ────────────────────────────────────────────────────────────────

  val weights: Any = cellType match {
    case Vanilla => VanillaWeights(GateWeights.random(hiddenDim, inputDim, rng))
    case LSTM    => LstmWeights.random(hiddenDim, inputDim, rng)
    case GRU     => GruWeights.random(hiddenDim, inputDim, rng)
  }

  // ── Forward pass ──────────────────────────────────────────────────────────

  /**
   * Runs the cell forward over a sequence.
   *
   * @param xs        sequence of input vectors (T x inputDim)
   * @param initState initial cell state
   * @param dropout   dropout rate (0 = no dropout)
   * @param training  if false, no dropout applied (inference mode)
   * @return (outputs: T x hiddenDim matrix, finalState, caches)
   */
  def forward(
    xs:        Array[DenseVector[Double]],
    initState: CellState,
    dropout:   Double,
    training:  Boolean
  ): (Array[DenseVector[Double]], CellState, Any) = {
    val T       = xs.length
    val outputs = new Array[DenseVector[Double]](T)
    val caches  = new Array[Any](T)
    var state   = initState

    val mask = if (dropout > 0 && training)
      Some(dropoutMask(hiddenDim, dropout))
    else if (dropout > 0 && !training)
      // MC dropout: apply mask even at inference
      Some(dropoutMask(hiddenDim, dropout))
    else
      None

    for (t <- 0 until T) {
      cellType match {
        case Vanilla =>
          val (newState, cache) = VanillaCell.forward(
            xs(t), state.asInstanceOf[VanillaState],
            weights.asInstanceOf[VanillaWeights], mask)
          outputs(t) = newState.h
          caches(t)  = cache
          state      = newState

        case LSTM =>
          val (newState, cache) = LstmCell.forward(
            xs(t), state.asInstanceOf[LstmState],
            weights.asInstanceOf[LstmWeights], mask)
          outputs(t) = newState.h
          caches(t)  = cache
          state      = newState

        case GRU =>
          val (newState, cache) = GruCell.forward(
            xs(t), state.asInstanceOf[GruState],
            weights.asInstanceOf[GruWeights], mask)
          outputs(t) = newState.h
          caches(t)  = cache
          state      = newState
      }
    }

    (outputs, state, caches)
  }

  // ── Backward pass (BPTT) ──────────────────────────────────────────────────

  /**
   * Backpropagates through time given upstream gradients on outputs.
   *
   * @param dOutputs  upstream gradients (T x hiddenDim)
   * @param caches    forward pass caches
   * @param dropout   dropout rate (must match forward pass)
   * @return (dInputs: T x inputDim, accumulated weight gradients)
   */
  def backward(
    dOutputs: Array[DenseVector[Double]],
    caches:   Any,
    dropout:  Double
  ): (Array[DenseVector[Double]], Any) = {
    val cachesArr = caches.asInstanceOf[Array[Any]]
    val T         = dOutputs.length
    val dInputs   = new Array[DenseVector[Double]](T)

    // We don't store the mask between forward and backward — use no mask here
    // since the mask is already baked into the cached states
    val mask: Option[DenseVector[Double]] = None

    cellType match {
      case Vanilla =>
        var dh    = DenseVector.zeros[Double](hiddenDim)
        val gradsAcc = VanillaGrads(GateGrads.zeros(hiddenDim, inputDim))
        val w     = weights.asInstanceOf[VanillaWeights]

        for (t <- (0 until T).reverse) {
          val dhTotal = dOutputs(t) + dh
          val (dx, dhPrev, g) = VanillaCell.backward(
            dhTotal, cachesArr(t).asInstanceOf[VanillaCache], w, mask)
          dInputs(t) = dx
          dh = dhPrev
          // accumulate
          addGateGrads(gradsAcc.gate, g.gate)
        }
        (dInputs, gradsAcc)

      case LSTM =>
        var dh    = DenseVector.zeros[Double](hiddenDim)
        var dc    = DenseVector.zeros[Double](hiddenDim)
        val gradsAcc = LstmGrads.zeros(hiddenDim, inputDim)
        val w     = weights.asInstanceOf[LstmWeights]

        for (t <- (0 until T).reverse) {
          val dhTotal = dOutputs(t) + dh
          val (dx, dhPrev, dcPrev, g) = LstmCell.backward(
            dhTotal, dc, cachesArr(t).asInstanceOf[LstmCache], w, mask)
          dInputs(t) = dx
          dh = dhPrev
          dc = dcPrev
          addLstmGrads(gradsAcc, g)
        }
        (dInputs, gradsAcc)

      case GRU =>
        var dh    = DenseVector.zeros[Double](hiddenDim)
        val gradsAcc = GruGrads.zeros(hiddenDim, inputDim)
        val w     = weights.asInstanceOf[GruWeights]

        for (t <- (0 until T).reverse) {
          val dhTotal = dOutputs(t) + dh
          val (dx, dhPrev, g) = GruCell.backward(
            dhTotal, cachesArr(t).asInstanceOf[GruCache], w, mask)
          dInputs(t) = dx
          dh = dhPrev
          addGruGrads(gradsAcc, g)
        }
        (dInputs, gradsAcc)
    }
  }

  // ── Gradient accumulation helpers ─────────────────────────────────────────

  private def addGateGrads(acc: GateGrads, g: GateGrads): Unit = {
    acc.dWx += g.dWx; acc.dWh += g.dWh; acc.db += g.db
  }

  private def addLstmGrads(acc: LstmGrads, g: LstmGrads): Unit = {
    addGateGrads(acc.wf, g.wf); addGateGrads(acc.wi, g.wi)
    addGateGrads(acc.wg, g.wg); addGateGrads(acc.wo, g.wo)
  }

  private def addGruGrads(acc: GruGrads, g: GruGrads): Unit = {
    addGateGrads(acc.wr, g.wr); addGateGrads(acc.wz, g.wz)
    addGateGrads(acc.wn, g.wn)
  }

  // ── Utilities ─────────────────────────────────────────────────────────────

  private def dropoutMask(dim: Int, rate: Double): DenseVector[Double] = {
    val scale = 1.0 / (1.0 - rate)  // inverted dropout
    DenseVector(Array.fill(dim)(
      if (rng.nextDouble() > rate) scale else 0.0
    ))
  }
}
