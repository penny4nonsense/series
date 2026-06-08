package series.rnn

import breeze.linalg._

/**
 * Fitted RNN model results.
 *
 * Provides point forecasts (recursive multi-step) and
 * MC dropout prediction intervals.
 */
case class RnnFit(
  net:        StackedRnn,
  normalizer: RnnNormalizer,
  config:     RnnConfig,
  window:     Int,
  nSeries:    Int,
  history:    TrainingHistory,
  nObs:       Int
) {

  /**
   * Point forecast for h steps ahead.
   * Uses the last `window` observations as the initial context.
   *
   * @param Y  full data matrix (n x k) — same series used for training
   * @param h  forecast horizon (defaults to config.horizon)
   * @return forecast matrix (h x k) in original scale
   */
  def forecast(Y: DenseMatrix[Double], h: Int = -1): DenseMatrix[Double] = {
    val horizon = if (h < 0) config.horizon else h
    val Ynorm   = normalizer.transform(Y)
    val context = Ynorm(math.max(0, Y.rows - window) until Y.rows, ::).copy

    val preds = DenseMatrix.zeros[Double](horizon, nSeries)
    var win   = context

    for (i <- 0 until horizon) {
      val (pred, _, _) = net.forward(win, dropout = 0.0, training = false)
      preds(i, ::) := pred.t

      // Slide window: drop first row, append prediction
      val predRow = pred.toDenseMatrix  // 1 x k
      win = DenseMatrix.vertcat(win(1 until win.rows, ::), predRow)
    }

    normalizer.inverseTransform(preds)
  }

  /**
   * MC dropout forecast with prediction intervals.
   *
   * Runs `nSamples` stochastic forward passes with dropout active.
   * Returns mean, lower and upper quantile bounds.
   *
   * @param Y        full data matrix
   * @param h        forecast horizon
   * @param level    confidence level (e.g. 0.95)
   * @param nSamples number of MC samples (overrides config.mcSamples if > 0)
   * @return RnnForecast with mean, lower, upper (all h x k in original scale)
   */
  def forecastWithIntervals(
    Y:        DenseMatrix[Double],
    h:        Int    = -1,
    level:    Double = 0.95,
    nSamples: Int    = -1
  ): RnnForecast = {
    val horizon  = if (h < 0) config.horizon else h
    val samples  = if (nSamples < 0) config.mcSamples else nSamples
    val alpha    = 1.0 - level
    val Ynorm    = normalizer.transform(Y)
    val context  = Ynorm(math.max(0, Y.rows - window) until Y.rows, ::).copy

    // Collect MC samples (samples x horizon x k)
    val allPreds = Array.tabulate(samples) { _ =>
      val win = context.copy
      val ps  = DenseMatrix.zeros[Double](horizon, nSeries)
      var w   = win
      for (i <- 0 until horizon) {
        // MC dropout: training=false but dropout > 0 means mask is applied
        val (pred, _, _) = net.forward(w, config.dropout, training = false)
        ps(i, ::) := pred.t
        val predRow = pred.toDenseMatrix
        w = DenseMatrix.vertcat(w(1 until w.rows, ::), predRow)
      }
      normalizer.inverseTransform(ps)
    }

    // Compute mean, lower quantile, upper quantile
    val mean  = DenseMatrix.zeros[Double](horizon, nSeries)
    val lower = DenseMatrix.zeros[Double](horizon, nSeries)
    val upper = DenseMatrix.zeros[Double](horizon, nSeries)

    for (i <- 0 until horizon; j <- 0 until nSeries) {
      val vals = allPreds.map(_(i, j)).sorted
      mean(i, j)  = vals.sum / samples
      lower(i, j) = vals((alpha / 2 * samples).toInt)
      upper(i, j) = vals(((1 - alpha / 2) * samples).toInt.min(samples - 1))
    }

    RnnForecast(mean = mean, lower = lower, upper = upper,
      h = horizon, k = nSeries, level = level, nSamples = samples)
  }

  /**
   * One-step-ahead in-sample fitted values.
   */
  def fitted(Y: DenseMatrix[Double]): DenseMatrix[Double] = {
    val Ynorm = normalizer.transform(Y)
    val wins  = WindowLoader.windows(Ynorm, window, config.horizon)
    val fits  = DenseMatrix.zeros[Double](wins.length, nSeries)
    for ((w, i) <- wins.zipWithIndex) {
      val (pred, _, _) = net.forward(w._1, 0.0, training = false)
      fits(i, ::) := pred.t
    }
    normalizer.inverseTransform(fits)
  }

  override def toString: String = {
    val sb = new StringBuilder
    sb.append(s"RNN Fit (${config.cellType}, layers=${config.nLayers}, " +
      s"hiddenDim=${config.hiddenDim})\n")
    sb.append(s"  Series: $nSeries, Window: $window, Horizon: ${config.horizon}\n")
    sb.append(f"  Final train loss: ${history.trainLosses.last}%.6f\n")
    if (history.valLosses.nonEmpty)
      sb.append(f"  Best val loss:   ${history.valLosses(history.bestEpoch)}%.6f " +
        s"(epoch ${history.bestEpoch})\n")
    sb.append(s"  Early stopped: ${history.stopped}\n")
    sb.toString
  }
}

/**
 * RNN forecast result with prediction intervals.
 */
case class RnnForecast(
  mean:     DenseMatrix[Double],
  lower:    DenseMatrix[Double],
  upper:    DenseMatrix[Double],
  h:        Int,
  k:        Int,
  level:    Double,
  nSamples: Int
) {
  override def toString: String = {
    val sb = new StringBuilder
    sb.append(s"RNN Forecast (h=$h, k=$k, ${(level*100).toInt}% PI, MC samples=$nSamples)\n")
    sb.append(s"  h" + (0 until k).map(j => s"  y${j+1}_mean  y${j+1}_lo  y${j+1}_hi").mkString + "\n")
    for (i <- 0 until h)
      sb.append(f"  ${i+1}%3d" + (0 until k).map(j =>
        f"  ${mean(i,j)}%9.4f  ${lower(i,j)}%9.4f  ${upper(i,j)}%9.4f").mkString + "\n")
    sb.toString
  }
}
