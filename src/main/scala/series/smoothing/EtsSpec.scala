package series.smoothing

import breeze.linalg._

/**
 * ETS (Error-Trend-Seasonal) state space model specification.
 *
 * The 15 admissible models from Hyndman et al. (2008):
 *   Error:   A (additive), M (multiplicative)
 *   Trend:   N (none), A (additive), Ad (additive damped)
 *   Seasonal: N (none), A (additive), M (multiplicative)
 *
 * Multiplicative seasonal with additive error is excluded as inadmissible.
 * MNM, MAM, MAdM are included (M error + M seasonal is admissible).
 *
 * State space form (additive error, general):
 *   y_t = l_{t-1} + b_{t-1} + s_{t-m} + alpha * e_t
 *   l_t = l_{t-1} + b_{t-1} + alpha * e_t
 *   b_t = phi*b_{t-1} + beta * e_t
 *   s_t = s_{t-m} + gamma * e_t
 *
 * where e_t ~ N(0, sigma^2), phi is damping parameter (1 for undamped).
 *
 * For multiplicative error: replace e_t with mu_t * e_t where
 * mu_t is the one-step-ahead mean.
 */

sealed trait ErrorType
case object AdditiveError      extends ErrorType
case object MultiplicativeError extends ErrorType

sealed trait TrendType
case object NoTrend        extends TrendType
case object AdditiveTrend  extends TrendType
case object DampedTrend    extends TrendType

sealed trait SeasonalType
case object NoSeasonal        extends SeasonalType
case object AdditiveSeasonal  extends SeasonalType
case object MultiplicativeSeasonal extends SeasonalType

/**
 * ETS model specification.
 *
 * @param error    error type
 * @param trend    trend type
 * @param seasonal seasonal type
 * @param period   seasonal period (ignored if NoSeasonal)
 */
case class EtsSpec(
  error:    ErrorType,
  trend:    TrendType,
  seasonal: SeasonalType,
  period:   Int = 1
) {
  val hasTrend:    Boolean = trend    != NoTrend
  val hasSeasonal: Boolean = seasonal != NoSeasonal
  val isDamped:    Boolean = trend    == DampedTrend

  /** Dimension of state vector [level, (trend), (seasonal lags)] */
  val stateDim: Int = 1 +
    (if (hasTrend) 1 else 0) +
    (if (hasSeasonal) period else 0)

  /** Number of smoothing parameters */
  val nParams: Int = {
    val base = 1 +  // alpha
      (if (hasTrend) 1 else 0) +     // beta
      (if (hasSeasonal) 1 else 0) +  // gamma
      (if (isDamped) 1 else 0)       // phi
    base + 1  // + sigma2
  }

  /** Human-readable name */
  def name: String = {
    val e = error match {
      case AdditiveError       => "A"
      case MultiplicativeError => "M"
    }
    val t = trend match {
      case NoTrend       => "N"
      case AdditiveTrend => "A"
      case DampedTrend   => "Ad"
    }
    val s = seasonal match {
      case NoSeasonal             => "N"
      case AdditiveSeasonal       => "A"
      case MultiplicativeSeasonal => "M"
    }
    s"ETS($e,$t,$s)"
  }

  override def toString: String = name
}

object EtsSpec {
  /** All 15 admissible ETS models */
  def allAdmissible(period: Int): Seq[EtsSpec] = Seq(
    // Additive error (9 models)
    EtsSpec(AdditiveError, NoTrend,       NoSeasonal,             period),
    EtsSpec(AdditiveError, AdditiveTrend, NoSeasonal,             period),
    EtsSpec(AdditiveError, DampedTrend,   NoSeasonal,             period),
    EtsSpec(AdditiveError, NoTrend,       AdditiveSeasonal,       period),
    EtsSpec(AdditiveError, AdditiveTrend, AdditiveSeasonal,       period),
    EtsSpec(AdditiveError, DampedTrend,   AdditiveSeasonal,       period),
    EtsSpec(AdditiveError, NoTrend,       MultiplicativeSeasonal, period),
    EtsSpec(AdditiveError, AdditiveTrend, MultiplicativeSeasonal, period),
    EtsSpec(AdditiveError, DampedTrend,   MultiplicativeSeasonal, period),
    // Multiplicative error (6 models — only non-additive-seasonal variants)
    EtsSpec(MultiplicativeError, NoTrend,       NoSeasonal,             period),
    EtsSpec(MultiplicativeError, AdditiveTrend, NoSeasonal,             period),
    EtsSpec(MultiplicativeError, DampedTrend,   NoSeasonal,             period),
    EtsSpec(MultiplicativeError, NoTrend,       MultiplicativeSeasonal, period),
    EtsSpec(MultiplicativeError, AdditiveTrend, MultiplicativeSeasonal, period),
    EtsSpec(MultiplicativeError, DampedTrend,   MultiplicativeSeasonal, period)
  )
}

/**
 * ETS state space matrices for a given specification.
 * Builds the Kalman filter representation.
 *
 * State vector layout:
 *   x = [l_t, b_t (if trend), s_t, s_{t-1}, ..., s_{t-m+1} (if seasonal)]
 *
 * For additive error:
 *   y_t = w' x_{t-1} + eps_t
 *   x_t = F x_{t-1} + g * eps_t
 *
 * where eps_t ~ N(0, sigma^2).
 */
class EtsStateSpace(spec: EtsSpec, params: EtsParams) {
  import spec._

  val m: Int = if (hasSeasonal) period else 1

  /** Measurement vector w (stateDim x 1) */
  def w(state: DenseVector[Double]): Double = {
    // w' x gives the conditional mean
    var mu = state(0)  // level
    if (hasTrend) mu += params.phi * state(1)
    if (hasSeasonal) {
      val sIdx = 1 + (if (hasTrend) 1 else 0)
      seasonal match {
        case AdditiveSeasonal       => mu += state(sIdx)
        case MultiplicativeSeasonal => mu *= state(sIdx)
        case NoSeasonal             =>
      }
    }
    mu
  }

  /** Transition: propagate state forward */
  def transition(state: DenseVector[Double], err: Double): DenseVector[Double] = {
    val newState = DenseVector.zeros[Double](stateDim)

    val l = state(0)
    val b = if (hasTrend) state(1) else 0.0
    val s = if (hasSeasonal) {
      val sIdx = 1 + (if (hasTrend) 1 else 0)
      state(sIdx)
    } else 0.0

    val mu = w(state)

    // Scale error for multiplicative vs additive
    val e = error match {
      case AdditiveError       => err
      case MultiplicativeError => mu * err
    }

    // Level update
    newState(0) = l + params.phi * b + params.alpha * e

    // Trend update
    if (hasTrend) {
      newState(1) = params.phi * b + params.beta * e
    }

    // Seasonal update
    if (hasSeasonal) {
      val sIdx = 1 + (if (hasTrend) 1 else 0)
      val sUpdate = seasonal match {
        case AdditiveSeasonal       => s + params.gamma * e
        case MultiplicativeSeasonal => s + params.gamma * e / math.max(math.abs(l + params.phi*b), 1e-10)
        case NoSeasonal             => 0.0
      }
      newState(sIdx) = sUpdate
      // Shift seasonal lags
      for (j <- sIdx + 1 until stateDim)
        newState(j) = state(j - 1)
    }

    newState
  }
}

/**
 * ETS smoothing parameters.
 *
 * @param alpha  level smoothing (0 < alpha < 1)
 * @param beta   trend smoothing (0 < beta < alpha; 0 if no trend)
 * @param gamma  seasonal smoothing (0 < gamma < 1-alpha; 0 if no seasonal)
 * @param phi    damping parameter (0.8 < phi < 0.98; 1 if not damped)
 * @param sigma2 innovation variance
 */
case class EtsParams(
  alpha:  Double,
  beta:   Double  = 0.0,
  gamma:  Double  = 0.0,
  phi:    Double  = 1.0,
  sigma2: Double  = 1.0
) {
  def toArray: Array[Double] = Array(alpha, beta, gamma, phi, sigma2)
}

object EtsParams {
  /** Default starting values for optimization */
  def defaults(spec: EtsSpec): EtsParams = EtsParams(
    alpha  = 0.1,
    beta   = if (spec.hasTrend) 0.01 else 0.0,
    gamma  = if (spec.hasSeasonal) 0.01 else 0.0,
    phi    = if (spec.isDamped) 0.98 else 1.0,
    sigma2 = 1.0
  )
}
