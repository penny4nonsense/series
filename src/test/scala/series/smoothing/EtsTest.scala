package series.smoothing

import org.scalatest.funsuite.AnyFunSuite

class EtsTest extends AnyFunSuite {

  val rng = new scala.util.Random(42)

  def arSeries(n: Int, phi: Double = 0.8): Array[Double] = {
    val y = new Array[Double](n)
    for (t <- 1 until n) y(t) = phi * y(t-1) + rng.nextGaussian() * 0.5
    y
  }

  def trendSeries(n: Int): Array[Double] =
    Array.tabulate(n)(t => 5.0 + 0.1 * t + rng.nextGaussian() * 0.3)

  def seasonalSeries(n: Int, period: Int = 12): Array[Double] =
    Array.tabulate(n) { t =>
      10.0 + 0.05*t + 2*math.sin(2*math.Pi*t/period) + rng.nextGaussian()*0.2
    }

  // ── EtsSpec ────────────────────────────────────────────────────────────────

  test("ETS(A,N,N) name is correct") {
    assert(EtsSpec(AdditiveError, NoTrend, NoSeasonal).name == "ETS(A,N,N)")
  }

  test("ETS(M,Ad,M) name is correct") {
    assert(EtsSpec(MultiplicativeError, DampedTrend, MultiplicativeSeasonal, 12).name == "ETS(M,Ad,M)")
  }

  test("15 admissible models for period=12") {
    assert(EtsSpec.allAdmissible(12).length == 15)
  }

  test("6 non-seasonal admissible models") {
    assert(EtsSpec.allAdmissible(1).count(_.seasonal == NoSeasonal) == 6)
  }

  test("state dimension ANN = 1") {
    assert(EtsSpec(AdditiveError, NoTrend, NoSeasonal).stateDim == 1)
  }

  test("state dimension AAN = 2") {
    assert(EtsSpec(AdditiveError, AdditiveTrend, NoSeasonal).stateDim == 2)
  }

  test("state dimension ANA with period 12 = 13") {
    assert(EtsSpec(AdditiveError, NoTrend, AdditiveSeasonal, 12).stateDim == 13)
  }

  // ── EtsEstimator ──────────────────────────────────────────────────────────

  test("ANN fit returns finite log-likelihood") {
    val y   = arSeries(100)
    val fit = new EtsEstimator(EtsSpec(AdditiveError, NoTrend, NoSeasonal)).fit(y)
    assert(!fit.logLik.isNaN && !fit.logLik.isInfinite)
  }

  test("AAN fit returns finite log-likelihood") {
    val y   = trendSeries(100)
    val fit = new EtsEstimator(EtsSpec(AdditiveError, AdditiveTrend, NoSeasonal)).fit(y)
    assert(!fit.logLik.isNaN)
  }

  test("alpha is in (0,1)") {
    val y   = arSeries(100)
    val fit = new EtsEstimator(EtsSpec(AdditiveError, NoTrend, NoSeasonal)).fit(y)
    assert(fit.params.alpha > 0 && fit.params.alpha < 1)
  }

  test("sigma2 is positive") {
    val y   = arSeries(100)
    val fit = new EtsEstimator(EtsSpec(AdditiveError, NoTrend, NoSeasonal)).fit(y)
    assert(fit.params.sigma2 > 0)
  }

  test("fitted values have correct length") {
    val y   = arSeries(100)
    val fit = new EtsEstimator(EtsSpec(AdditiveError, NoTrend, NoSeasonal)).fit(y)
    assert(fit.fitted.length == 100)
  }

  test("residuals have correct length") {
    val y   = arSeries(100)
    val fit = new EtsEstimator(EtsSpec(AdditiveError, NoTrend, NoSeasonal)).fit(y)
    assert(fit.residuals.length == 100)
  }

  test("ANN with alpha=1 fitted values approximate naive forecast") {
    val y    = arSeries(50)
    val spec = EtsSpec(AdditiveError, NoTrend, NoSeasonal)
    val est  = new EtsEstimator(spec)
    val params = EtsParams(alpha=0.999, sigma2=1.0)
    val init   = est.initializeState(y)
    // alpha ~ 1 means y_hat(t) ~ y(t-1)
    val ss = new EtsStateSpace(spec, params)
    var x  = init.copy
    for (t <- 0 until 5) {
      val pred = ss.w(x)
      assert(math.abs(pred - y(math.max(0, t-1))) < 0.5)
      val err = y(t) - pred
      x = ss.transition(x, err / math.sqrt(params.sigma2))
    }
  }

  // ── EtsForecaster ─────────────────────────────────────────────────────────

  test("forecast has correct length") {
    val y    = arSeries(100)
    val fit  = new EtsEstimator(EtsSpec(AdditiveError, NoTrend, NoSeasonal)).fit(y)
    val fc   = new EtsForecaster(fit).forecast(10)
    assert(fc.means.length == 10)
  }

  test("forecast intervals are ordered") {
    val y  = arSeries(100)
    val fit = new EtsEstimator(EtsSpec(AdditiveError, NoTrend, NoSeasonal)).fit(y)
    val fc  = new EtsForecaster(fit).forecast(10)
    fc.lower.zip(fc.upper).foreach { case (l, u) => assert(l <= u) }
  }

  test("forecast values are finite") {
    val y   = arSeries(100)
    val fit = new EtsEstimator(EtsSpec(AdditiveError, NoTrend, NoSeasonal)).fit(y)
    val fc  = new EtsForecaster(fit).forecast(10)
    assert(fc.means.forall(v => !v.isNaN && !v.isInfinite))
  }

  // ── AutoEts ───────────────────────────────────────────────────────────────

  test("AutoEts selects a model") {
    val y = arSeries(100)
    val r = new AutoEts(1).fit(y)
    assert(r.best.spec.name.startsWith("ETS"))
  }

  test("AutoEts returns multiple models") {
    val y = arSeries(100)
    val r = new AutoEts(1).fit(y)
    assert(r.nModels >= 3)
  }

  test("AutoEts best model has lowest AICc") {
    val y = arSeries(100)
    val r = new AutoEts(1).fit(y)
    assert(r.best.aicc <= r.fits.map(_.aicc).min + 1e-8)
  }

  test("AutoEts toString is readable") {
    val y = arSeries(100)
    val r = new AutoEts(1).fit(y)
    assert(r.toString.contains("ETS"))
  }

  test("seasonal AutoEts fits seasonal model") {
    // Use a longer series with a strong seasonal signal and no noise
    val y = Array.tabulate(240) { t =>
      10.0 + 0.02*t + 4*math.sin(2*math.Pi*t/12)
    }
    val r = new AutoEts(12).fit(y)
    assert(r.best.spec.hasSeasonal)
  }
}
