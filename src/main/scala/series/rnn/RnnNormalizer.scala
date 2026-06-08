package series.rnn

import breeze.linalg._

/**
 * Per-series mean/std normalization for RNN inputs.
 *
 * Fit on training data only. Applied to both input windows and targets.
 * Inverse applied to forecasts.
 *
 * For a series with std near zero (constant), std is set to 1.0
 * to avoid division by zero.
 */
class RnnNormalizer(val nSeries: Int) {
  var means: Array[Double] = Array.fill(nSeries)(0.0)
  var stds:  Array[Double] = Array.fill(nSeries)(1.0)
  private var fitted = false

  /**
   * Fits normalizer parameters from training data.
   * @param Y matrix (n x k) — rows are time points, columns are series
   */
  def fit(Y: DenseMatrix[Double]): Unit = {
    require(Y.cols == nSeries, s"expected $nSeries series, got ${Y.cols}")
    val n = Y.rows.toDouble
    for (j <- 0 until nSeries) {
      val col  = Y(::, j)
      val mean = col.toArray.sum / n
      val std  = math.sqrt(col.toArray.map(v => math.pow(v - mean, 2)).sum / n)
      means(j) = mean
      stds(j)  = math.max(std, 1e-8)
    }
    fitted = true
  }

  /** Normalizes a matrix in place (returns new matrix) */
  def transform(Y: DenseMatrix[Double]): DenseMatrix[Double] = {
    require(fitted, "normalizer not yet fit")
    DenseMatrix.tabulate(Y.rows, Y.cols) { (i, j) =>
      (Y(i, j) - means(j)) / stds(j)
    }
  }

  /** Denormalizes a matrix */
  def inverseTransform(Y: DenseMatrix[Double]): DenseMatrix[Double] = {
    require(fitted, "normalizer not yet fit")
    DenseMatrix.tabulate(Y.rows, Y.cols) { (i, j) =>
      Y(i, j) * stds(j) + means(j)
    }
  }

  /** Normalize a single vector (one time step across all series) */
  def transformVec(v: DenseVector[Double]): DenseVector[Double] =
    DenseVector(v.toArray.zipWithIndex.map { case (x, j) => (x - means(j)) / stds(j) })

  /** Denormalize a single vector */
  def inverseTransformVec(v: DenseVector[Double]): DenseVector[Double] =
    DenseVector(v.toArray.zipWithIndex.map { case (x, j) => x * stds(j) + means(j) })
}
