package series.rnn

import breeze.linalg._

/**
 * Top-level RNN model API.
 *
 * Usage:
 *   val fit = RnnModel(RnnConfig.medium).fit(Y)
 *   val fc  = fit.forecast(h = 10)
 *   val pi  = fit.forecastWithIntervals(h = 10, level = 0.95)
 *
 * @param config model configuration
 */
class RnnModel(val config: RnnConfig) {

  /**
   * Fits the RNN to a multivariate time series.
   *
   * @param Y data matrix (n x k) — rows are time points, columns are series
   * @return RnnFit with trained weights and training history
   */
  def fit(Y: DenseMatrix[Double]): RnnFit = {
    val n   = Y.rows
    val k   = Y.cols
    val w   = config.effectiveWindow(n)
    val rng = new scala.util.Random(config.seed)

    require(n > w + config.horizon,
      s"series length $n too short for window $w + horizon ${config.horizon}")

    // Normalize
    val normalizer = new RnnNormalizer(k)
    val splitIdx   = math.max(w + config.horizon, (n * (1 - config.valFraction)).toInt)
    normalizer.fit(Y(0 until splitIdx, ::))
    val Ynorm = normalizer.transform(Y)

    // Build windows
    val allWindows = WindowLoader.windows(Ynorm, w, config.horizon)
    val nTrain     = (allWindows.length * (1 - config.valFraction)).toInt
    val trainWin   = allWindows.take(nTrain)
    val valWin     = allWindows.drop(nTrain)

    // Build stacked RNN
    val net = new StackedRnn(config, k, rng)

    val trainLosses = new Array[Double](config.epochs)
    val valLosses   = new Array[Double](config.epochs)
    var bestValLoss = Double.MaxValue
    var bestEpoch   = 0
    var patience    = 0
    var stopped     = false

    for (epoch <- 0 until config.epochs if !stopped) {
      // Training
      val shuffled = WindowLoader.shuffle(trainWin, rng)
      var epochLoss = 0.0
      var nBatches  = 0

      shuffled.grouped(config.batchSize).foreach { batch =>
        var batchLoss = 0.0
        for ((window, target) <- batch) {
          val (pred, caches, _) = net.forward(window, config.dropout, training = true)
          val (loss, dLoss)     = LossType.loss(pred, target, config.lossType)
          batchLoss += loss
          net.backward(dLoss, caches, window, config.dropout)
        }
        epochLoss += batchLoss / batch.length
        nBatches  += 1
      }

      trainLosses(epoch) = epochLoss / math.max(nBatches, 1)

      // Validation
      if (valWin.nonEmpty) {
        var vLoss = 0.0
        for ((window, target) <- valWin) {
          val (pred, _, _) = net.forward(window, config.dropout, training = false)
          val (loss, _)    = LossType.loss(pred, target, config.lossType)
          vLoss += loss
        }
        val avgVal = vLoss / valWin.length
        valLosses(epoch) = avgVal

        // Early stopping
        if (avgVal < bestValLoss - config.minDelta) {
          bestValLoss = avgVal
          bestEpoch   = epoch
          patience    = 0
        } else {
          patience += 1
          if (patience >= config.patience) stopped = true
        }
      }
    }

    val history = TrainingHistory(
      trainLosses = trainLosses.take(if (stopped) bestEpoch + config.patience + 1 else config.epochs),
      valLosses   = valLosses.take(if (stopped) bestEpoch + config.patience + 1 else config.epochs),
      bestEpoch   = bestEpoch,
      stopped     = stopped
    )

    RnnFit(net = net, normalizer = normalizer, config = config,
      window = w, nSeries = k, history = history, nObs = n)
  }
}

object RnnModel {
  def apply(config: RnnConfig = RnnConfig()): RnnModel = new RnnModel(config)
}
