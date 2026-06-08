package series.rnn

import breeze.linalg._

/**
 * Adam optimizer state for a single parameter tensor.
 */
case class AdamState(
  m:  DenseMatrix[Double],  // first moment
  v:  DenseMatrix[Double],  // second moment
  t:  Int                   // time step
)

object AdamState {
  def zeros(rows: Int, cols: Int): AdamState =
    AdamState(DenseMatrix.zeros(rows, cols), DenseMatrix.zeros(rows, cols), 0)
}

case class AdamStateVec(
  m: DenseVector[Double],
  v: DenseVector[Double],
  t: Int
)

object AdamStateVec {
  def zeros(dim: Int): AdamStateVec =
    AdamStateVec(DenseVector.zeros(dim), DenseVector.zeros(dim), 0)
}

/**
 * Adam update for a matrix parameter.
 * Returns (updated param, updated state).
 */
object Adam {
  val beta1  = 0.9
  val beta2  = 0.999
  val eps    = 1e-8

  def updateMatrix(
    param: DenseMatrix[Double],
    grad:  DenseMatrix[Double],
    state: AdamState,
    lr:    Double
  ): (DenseMatrix[Double], AdamState) = {
    val t1  = state.t + 1
    val m1  = state.m * beta1 + grad * (1 - beta1)
    val v1  = state.v *:* DenseMatrix.fill(grad.rows, grad.cols)(beta2) +
              (grad *:* grad) * (1 - beta2)
    val mHat = m1 / (1 - math.pow(beta1, t1))
    val vHat = v1 / (1 - math.pow(beta2, t1))
    val update = DenseMatrix.tabulate(grad.rows, grad.cols) { (i, j) =>
      lr * mHat(i, j) / (math.sqrt(vHat(i, j)) + eps)
    }
    (param - update, AdamState(m1, v1, t1))
  }

  def updateVector(
    param: DenseVector[Double],
    grad:  DenseVector[Double],
    state: AdamStateVec,
    lr:    Double
  ): (DenseVector[Double], AdamStateVec) = {
    val t1   = state.t + 1
    val m1   = state.m * beta1 + grad * (1 - beta1)
    val v1   = state.v *:* DenseVector.fill(grad.length)(beta2) +
               (grad *:* grad) * (1 - beta2)
    val mHat = m1 / (1 - math.pow(beta1, t1))
    val vHat = v1 / (1 - math.pow(beta2, t1))
    val update = DenseVector.tabulate(grad.length)(i =>
      lr * mHat(i) / (math.sqrt(vHat(i)) + eps))
    (param - update, AdamStateVec(m1, v1, t1))
  }
}

/**
 * Gradient clipping by global norm.
 * Scales all gradients so the global L2 norm <= clipValue.
 */
object GradientClipper {
  def clipNorm(grads: Seq[DenseMatrix[Double]], clipValue: Double): Double = {
    val globalNorm = math.sqrt(grads.map(g => breeze.linalg.sum(g *:* g)).sum)
    if (globalNorm > clipValue) clipValue / math.max(globalNorm, 1e-10)
    else 1.0
  }
}

/**
 * Sliding window data loader.
 * Creates (input window, target) pairs from a multivariate time series.
 */
object WindowLoader {
  /**
   * @param Y        normalized time series (n x k)
   * @param window   lookback window
   * @param horizon  forecast horizon
   * @return sequence of (input: window x k, target: k) pairs
   */
  def windows(
    Y:       DenseMatrix[Double],
    window:  Int,
    horizon: Int
  ): Array[(DenseMatrix[Double], DenseVector[Double])] = {
    val n = Y.rows
    require(n > window + horizon, "series too short for window + horizon")
    (0 until n - window - horizon + 1).map { i =>
      val input  = Y(i until i + window, ::).copy
      val target = Y(i + window + horizon - 1, ::).t.copy
      (input, target)
    }.toArray
  }

  /** Shuffle windows (for minibatch SGD) */
  def shuffle[A](arr: Array[A], rng: scala.util.Random): Array[A] = {
    val a = arr.clone()
    for (i <- a.length - 1 to 1 by -1) {
      val j = rng.nextInt(i + 1)
      val tmp = a(i); a(i) = a(j); a(j) = tmp
    }
    a
  }
}

/**
 * Training loop result.
 */
case class TrainingHistory(
  trainLosses: Array[Double],
  valLosses:   Array[Double],
  bestEpoch:   Int,
  stopped:     Boolean
)
