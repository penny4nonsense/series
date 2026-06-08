package series.rnn

import breeze.linalg._

// ── Cell types ────────────────────────────────────────────────────────────────

sealed trait CellType
case object Vanilla extends CellType
case object LSTM    extends CellType
case object GRU     extends CellType

// ── Loss types ────────────────────────────────────────────────────────────────

sealed trait LossType
case object MSE   extends LossType
case object MAE   extends LossType
case object Huber extends LossType

object LossType {
  /** Compute scalar loss and gradient w.r.t. prediction */
  def loss(pred: DenseVector[Double], target: DenseVector[Double],
           lossType: LossType, delta: Double = 1.0): (Double, DenseVector[Double]) = {
    val diff = pred - target
    lossType match {
      case MSE =>
        val k   = diff.length.toDouble
        val l   = (diff dot diff) / k
        val g   = diff * (2.0 / k)
        (l, g)

      case MAE =>
        val k = diff.length.toDouble
        val l = diff.toArray.map(math.abs).sum / k
        val g = DenseVector(diff.toArray.map(d => math.signum(d) / k))
        (l, g)

      case Huber =>
        val k = diff.length.toDouble
        var l = 0.0
        val g = new Array[Double](diff.length)
        for (i <- 0 until diff.length) {
          val d = diff(i)
          if (math.abs(d) <= delta) {
            l    += 0.5 * d * d
            g(i)  = d
          } else {
            l    += delta * (math.abs(d) - 0.5 * delta)
            g(i)  = delta * math.signum(d)
          }
        }
        (l / k, DenseVector(g.map(_ / k)))
    }
  }
}

// ── Cell state ────────────────────────────────────────────────────────────────

sealed trait CellState {
  def h: DenseVector[Double]  // hidden state (output)
}

case class VanillaState(h: DenseVector[Double]) extends CellState

case class LstmState(
  h: DenseVector[Double],  // hidden state
  c: DenseVector[Double]   // cell state
) extends CellState

case class GruState(h: DenseVector[Double]) extends CellState

object CellState {
  def zeros(cellType: CellType, hiddenDim: Int): CellState = cellType match {
    case Vanilla => VanillaState(DenseVector.zeros[Double](hiddenDim))
    case LSTM    => LstmState(DenseVector.zeros[Double](hiddenDim),
                               DenseVector.zeros[Double](hiddenDim))
    case GRU     => GruState(DenseVector.zeros[Double](hiddenDim))
  }
}

// ── RNN configuration ─────────────────────────────────────────────────────────

/**
 * Configuration for an RNN model.
 *
 * Defaults are conservative safe minimums suitable for quick exploration.
 * For most real problems, use RnnConfig.medium or larger, or tune manually.
 *
 * Recommended ranges:
 *   hiddenDim:    64-256  (larger for complex patterns)
 *   nLayers:      1-3     (deeper for hierarchical patterns)
 *   epochs:       200-500 (more for convergence on small data)
 *   dropout:      0.1-0.3 (more for regularization on small data)
 *   window:       1-2 seasonal periods (or 10-50 for non-seasonal)
 *
 * @param cellType     RNN cell architecture
 * @param hiddenDim    hidden state dimension per layer
 * @param nLayers      number of stacked RNN layers
 * @param window       lookback window (-1 = auto)
 * @param horizon      forecast horizon (steps ahead)
 * @param period       seasonal period (used for auto window calculation)
 * @param dropout      dropout rate for regularization and MC uncertainty
 * @param lossType     training loss function
 * @param epochs       maximum training epochs
 * @param batchSize    minibatch size
 * @param learningRate Adam learning rate
 * @param patience     early stopping patience (epochs without improvement)
 * @param minDelta     minimum improvement for early stopping
 * @param clipValue    gradient clipping threshold
 * @param valFraction  fraction of data held out for validation
 * @param mcSamples    MC dropout samples for prediction intervals
 * @param seed         random seed
 */
case class RnnConfig(
  cellType:     CellType = LSTM,
  hiddenDim:    Int      = 32,
  nLayers:      Int      = 1,
  window:       Int      = -1,
  horizon:      Int      = 1,
  period:       Int      = 1,
  dropout:      Double   = 0.1,
  lossType:     LossType = MSE,
  epochs:       Int      = 100,
  batchSize:    Int      = 32,
  learningRate: Double   = 0.001,
  patience:     Int      = 10,
  minDelta:     Double   = 1e-4,
  clipValue:    Double   = 5.0,
  valFraction:  Double   = 0.2,
  mcSamples:    Int      = 200,
  seed:         Long     = 42L
) {
  require(hiddenDim    >= 1,   "hiddenDim must be positive")
  require(nLayers      >= 1,   "nLayers must be positive")
  require(horizon      >= 1,   "horizon must be positive")
  require(period       >= 1,   "period must be positive")
  require(dropout      >= 0.0 && dropout < 1.0, "dropout must be in [0,1)")
  require(epochs       >= 1,   "epochs must be positive")
  require(batchSize    >= 1,   "batchSize must be positive")
  require(learningRate  > 0,   "learningRate must be positive")
  require(patience     >= 1,   "patience must be positive")
  require(clipValue     > 0,   "clipValue must be positive")
  require(valFraction   > 0 && valFraction < 1, "valFraction must be in (0,1)")
  require(mcSamples    >= 1,   "mcSamples must be positive")

  /** Effective window size given data length n */
  def effectiveWindow(n: Int): Int = {
    if (window > 0) window
    else {
      val periodBased  = if (period > 1) 2 * period else 10
      val horizonBased = math.max(horizon * 2, periodBased)
      math.max(5, math.min(horizonBased, n / 3))
    }
  }
}

object RnnConfig {
  /** Conservative defaults — fast to run, may underfit */
  def small: RnnConfig = RnnConfig()

  /** Reasonable for most time series problems */
  def medium: RnnConfig = RnnConfig(
    hiddenDim    = 64,
    nLayers      = 2,
    epochs       = 200,
    dropout      = 0.2,
    patience     = 15
  )

  /** For complex patterns with sufficient data */
  def large: RnnConfig = RnnConfig(
    hiddenDim    = 128,
    nLayers      = 3,
    epochs       = 500,
    dropout      = 0.2,
    patience     = 20,
    batchSize    = 64
  )
}
