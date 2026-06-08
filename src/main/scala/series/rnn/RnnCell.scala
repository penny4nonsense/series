package series.rnn

import breeze.linalg._
import breeze.numerics._

/**
 * RNN cell implementations.
 *
 * Each cell takes:
 *   x     — input vector (inputDim)
 *   state — previous cell state
 *   dropMask — optional dropout mask (applied to hidden state)
 *
 * Returns:
 *   (output, newState, cache)
 *
 * The cache stores intermediate values needed for BPTT.
 */

// ── Activation utilities ──────────────────────────────────────────────────────

object Activations {
  @inline def tanhFn(x: Double): Double = math.tanh(x)
  @inline def tanhGrad(tanhVal: Double): Double = 1.0 - tanhVal * tanhVal
  @inline def sigmoidFn(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
  @inline def sigmoidGrad(sigVal: Double): Double = sigVal * (1.0 - sigVal)

  def tanhVec(v: DenseVector[Double]): DenseVector[Double] =
    DenseVector(v.toArray.map(tanhFn))

  def sigmoidVec(v: DenseVector[Double]): DenseVector[Double] =
    DenseVector(v.toArray.map(sigmoidFn))

  def tanhGradVec(tanhVals: DenseVector[Double]): DenseVector[Double] =
    DenseVector(tanhVals.toArray.map(tanhGrad))

  def sigmoidGradVec(sigVals: DenseVector[Double]): DenseVector[Double] =
    DenseVector(sigVals.toArray.map(sigmoidGrad))
}

// ── Cell weights ──────────────────────────────────────────────────────────────

/** Weights for a single gate or the vanilla RNN hidden layer */
case class GateWeights(
  Wx: DenseMatrix[Double],  // hiddenDim x inputDim
  Wh: DenseMatrix[Double],  // hiddenDim x hiddenDim
  b:  DenseVector[Double]   // hiddenDim
) {
  def preactivation(x: DenseVector[Double], h: DenseVector[Double]): DenseVector[Double] =
    (Wx * x) + (Wh * h) + b
}

object GateWeights {
  def random(hiddenDim: Int, inputDim: Int, rng: scala.util.Random,
             scale: Double = 0.1): GateWeights = {
    val Wx = DenseMatrix.tabulate(hiddenDim, inputDim)((_, _) => rng.nextGaussian() * scale)
    val Wh = DenseMatrix.tabulate(hiddenDim, hiddenDim)((_, _) => rng.nextGaussian() * scale)
    val b  = DenseVector.zeros[Double](hiddenDim)
    GateWeights(Wx, Wh, b)
  }

  def zeros(hiddenDim: Int, inputDim: Int): GateWeights =
    GateWeights(
      DenseMatrix.zeros[Double](hiddenDim, inputDim),
      DenseMatrix.zeros[Double](hiddenDim, hiddenDim),
      DenseVector.zeros[Double](hiddenDim)
    )
}

case class GateGrads(
  dWx: DenseMatrix[Double],
  dWh: DenseMatrix[Double],
  db:  DenseVector[Double]
)

object GateGrads {
  def zeros(hiddenDim: Int, inputDim: Int): GateGrads =
    GateGrads(
      DenseMatrix.zeros[Double](hiddenDim, inputDim),
      DenseMatrix.zeros[Double](hiddenDim, hiddenDim),
      DenseVector.zeros[Double](hiddenDim)
    )

  /** Accumulate gradients: dGate = dh_gate (*) grad, x, h_prev */
  def compute(
    dGate:  DenseVector[Double],
    x:      DenseVector[Double],
    hPrev:  DenseVector[Double]
  ): GateGrads = GateGrads(
    dWx = dGate.toDenseMatrix.t * x.toDenseMatrix,
    dWh = dGate.toDenseMatrix.t * hPrev.toDenseMatrix,
    db  = dGate
  )
}

// ── Vanilla RNN cell ──────────────────────────────────────────────────────────

case class VanillaWeights(gate: GateWeights)

case class VanillaGrads(gate: GateGrads)

/** Cache for vanilla RNN backward pass */
case class VanillaCache(
  x:     DenseVector[Double],
  hPrev: DenseVector[Double],
  z:     DenseVector[Double],   // pre-activation
  h:     DenseVector[Double]    // post-activation = tanh(z)
)

object VanillaCell {
  def forward(
    x:       DenseVector[Double],
    state:   VanillaState,
    weights: VanillaWeights,
    mask:    Option[DenseVector[Double]] = None
  ): (VanillaState, VanillaCache) = {
    import Activations._
    val z = weights.gate.preactivation(x, state.h)
    val h = tanhVec(z)
    val hDropped = mask match {
      case Some(m) => DenseVector(h.toArray.zip(m.toArray).map { case (hi, mi) => hi * mi })
      case None    => h
    }
    (VanillaState(hDropped), VanillaCache(x, state.h, z, h))
  }

  def backward(
    dh:      DenseVector[Double],
    cache:   VanillaCache,
    weights: VanillaWeights,
    mask:    Option[DenseVector[Double]] = None
  ): (DenseVector[Double], DenseVector[Double], VanillaGrads) = {
    import Activations._
    val dhMasked = mask match {
      case Some(m) => DenseVector(dh.toArray.zip(m.toArray).map { case (di, mi) => di * mi })
      case None    => dh
    }
    val dz    = dhMasked *:* tanhGradVec(cache.h)
    val dx    = weights.gate.Wx.t * dz
    val dhPrev = weights.gate.Wh.t * dz
    val grads = VanillaGrads(GateGrads.compute(dz, cache.x, cache.hPrev))
    (dx, dhPrev, grads)
  }
}

// ── LSTM cell ─────────────────────────────────────────────────────────────────

case class LstmWeights(
  wf: GateWeights,  // forget gate
  wi: GateWeights,  // input gate
  wg: GateWeights,  // cell gate (g)
  wo: GateWeights   // output gate
)

object LstmWeights {
  def random(hiddenDim: Int, inputDim: Int, rng: scala.util.Random): LstmWeights = {
    val scale = math.sqrt(2.0 / (hiddenDim + inputDim))  // Xavier
    LstmWeights(
      GateWeights.random(hiddenDim, inputDim, rng, scale),
      GateWeights.random(hiddenDim, inputDim, rng, scale),
      GateWeights.random(hiddenDim, inputDim, rng, scale),
      GateWeights.random(hiddenDim, inputDim, rng, scale)
    )
  }
}

case class LstmGrads(
  wf: GateGrads,
  wi: GateGrads,
  wg: GateGrads,
  wo: GateGrads
)

object LstmGrads {
  def zeros(hiddenDim: Int, inputDim: Int): LstmGrads = LstmGrads(
    GateGrads.zeros(hiddenDim, inputDim),
    GateGrads.zeros(hiddenDim, inputDim),
    GateGrads.zeros(hiddenDim, inputDim),
    GateGrads.zeros(hiddenDim, inputDim)
  )
}

case class LstmCache(
  x:     DenseVector[Double],
  hPrev: DenseVector[Double],
  cPrev: DenseVector[Double],
  f:     DenseVector[Double],   // forget gate
  i:     DenseVector[Double],   // input gate
  g:     DenseVector[Double],   // cell gate (tanh)
  o:     DenseVector[Double],   // output gate
  c:     DenseVector[Double],   // cell state
  tanhC: DenseVector[Double]    // tanh(c)
)

object LstmCell {
  def forward(
    x:       DenseVector[Double],
    state:   LstmState,
    weights: LstmWeights,
    mask:    Option[DenseVector[Double]] = None
  ): (LstmState, LstmCache) = {
    import Activations._
    val f     = sigmoidVec(weights.wf.preactivation(x, state.h))
    val i     = sigmoidVec(weights.wi.preactivation(x, state.h))
    val g     = tanhVec(weights.wg.preactivation(x, state.h))
    val o     = sigmoidVec(weights.wo.preactivation(x, state.h))
    // cell state computed below


    // c = f *:* cPrev + i *:* g
    val cNew  = DenseVector((0 until f.length).map(j =>
      f(j) * state.c(j) + i(j) * g(j)).toArray)
    val tanhC = tanhVec(cNew)
    // h = o *:* tanh(c)
    val hNew  = DenseVector((0 until o.length).map(j => o(j) * tanhC(j)).toArray)
    val hDropped = mask match {
      case Some(m) => DenseVector(hNew.toArray.zip(m.toArray).map { case (hi, mi) => hi * mi })
      case None    => hNew
    }
    (LstmState(hDropped, cNew), LstmCache(x, state.h, state.c, f, i, g, o, cNew, tanhC))
  }

  def backward(
    dh:      DenseVector[Double],
    dc:      DenseVector[Double],
    cache:   LstmCache,
    weights: LstmWeights,
    mask:    Option[DenseVector[Double]] = None
  ): (DenseVector[Double], DenseVector[Double], DenseVector[Double], LstmGrads) = {
    import Activations._
    val dhMasked = mask match {
      case Some(m) => DenseVector(dh.toArray.zip(m.toArray).map { case (di, mi) => di * mi })
      case None    => dh
    }

    // Gradients through output gate
    val do_   = DenseVector((0 until dh.length).map(j => dhMasked(j) * cache.tanhC(j)).toArray)
    val dcFromH = DenseVector((0 until dh.length).map(j =>
      dhMasked(j) * cache.o(j) * tanhGrad(cache.tanhC(j))).toArray)
    val dcTotal = dcFromH + dc

    // Gradients through cell state
    val df = DenseVector((0 until cache.f.length).map(j => dcTotal(j) * cache.cPrev(j)).toArray)
    val di = DenseVector((0 until cache.i.length).map(j => dcTotal(j) * cache.g(j)).toArray)
    val dg = DenseVector((0 until cache.g.length).map(j => dcTotal(j) * cache.i(j)).toArray)
    val dcPrev = DenseVector((0 until cache.f.length).map(j => dcTotal(j) * cache.f(j)).toArray)

    // Gate pre-activation gradients
    val dzo = do_ *:* sigmoidGradVec(cache.o)
    val dzf = df  *:* sigmoidGradVec(cache.f)
    val dzi = di  *:* sigmoidGradVec(cache.i)
    val dzg = dg  *:* tanhGradVec(cache.g)

    // Input and hidden gradients
    val dx     = weights.wo.Wx.t*dzo + weights.wf.Wx.t*dzf +
                 weights.wi.Wx.t*dzi + weights.wg.Wx.t*dzg
    val dhPrev = weights.wo.Wh.t*dzo + weights.wf.Wh.t*dzf +
                 weights.wi.Wh.t*dzi + weights.wg.Wh.t*dzg

    val grads = LstmGrads(
      wf = GateGrads.compute(dzf, cache.x, cache.hPrev),
      wi = GateGrads.compute(dzi, cache.x, cache.hPrev),
      wg = GateGrads.compute(dzg, cache.x, cache.hPrev),
      wo = GateGrads.compute(dzo, cache.x, cache.hPrev)
    )
    (dx, dhPrev, dcPrev, grads)
  }
}

// ── GRU cell ──────────────────────────────────────────────────────────────────

case class GruWeights(
  wr: GateWeights,  // reset gate
  wz: GateWeights,  // update gate
  wn: GateWeights   // new gate (candidate hidden state)
)

object GruWeights {
  def random(hiddenDim: Int, inputDim: Int, rng: scala.util.Random): GruWeights = {
    val scale = math.sqrt(2.0 / (hiddenDim + inputDim))
    GruWeights(
      GateWeights.random(hiddenDim, inputDim, rng, scale),
      GateWeights.random(hiddenDim, inputDim, rng, scale),
      GateWeights.random(hiddenDim, inputDim, rng, scale)
    )
  }
}

case class GruGrads(
  wr: GateGrads,
  wz: GateGrads,
  wn: GateGrads
)

object GruGrads {
  def zeros(hiddenDim: Int, inputDim: Int): GruGrads = GruGrads(
    GateGrads.zeros(hiddenDim, inputDim),
    GateGrads.zeros(hiddenDim, inputDim),
    GateGrads.zeros(hiddenDim, inputDim)
  )
}

case class GruCache(
  x:     DenseVector[Double],
  hPrev: DenseVector[Double],
  r:     DenseVector[Double],   // reset gate
  z:     DenseVector[Double],   // update gate
  n:     DenseVector[Double],   // candidate hidden state
  h:     DenseVector[Double]    // output hidden state
)

object GruCell {
  def forward(
    x:       DenseVector[Double],
    state:   GruState,
    weights: GruWeights,
    mask:    Option[DenseVector[Double]] = None
  ): (GruState, GruCache) = {
    import Activations._
    val r    = sigmoidVec(weights.wr.preactivation(x, state.h))
    val z    = sigmoidVec(weights.wz.preactivation(x, state.h))
    // n = tanh(Wn_x * x + Wn_h * (r *:* h_prev) + bn)
    val rh   = DenseVector(r.toArray.zip(state.h.toArray).map { case (ri, hi) => ri * hi })
    val nPre = (weights.wn.Wx * x) + (weights.wn.Wh * rh) + weights.wn.b
    val n    = tanhVec(nPre)
    // h = (1-z) *:* n + z *:* hPrev
    val hArr = (0 until z.length).map(j => (1.0 - z(j)) * n(j) + z(j) * state.h(j)).toArray
    val h    = DenseVector(hArr)
    val hDropped = mask match {
      case Some(m) => DenseVector(h.toArray.zip(m.toArray).map { case (hi, mi) => hi * mi })
      case None    => h
    }
    (GruState(hDropped), GruCache(x, state.h, r, z, n, h))
  }

  def backward(
    dh:      DenseVector[Double],
    cache:   GruCache,
    weights: GruWeights,
    mask:    Option[DenseVector[Double]] = None
  ): (DenseVector[Double], DenseVector[Double], GruGrads) = {
    import Activations._
    val dhMasked = mask match {
      case Some(m) => DenseVector(dh.toArray.zip(m.toArray).map { case (di, mi) => di * mi })
      case None    => dh
    }

    // dh/dz, dh/dn, dh/dhPrev
    val dn    = DenseVector((0 until dh.length).map(j => dhMasked(j) * (1.0 - cache.z(j))).toArray)
    val dz    = DenseVector((0 until dh.length).map(j =>
      dhMasked(j) * (cache.hPrev(j) - cache.n(j))).toArray)
    val dhPrevFromH = DenseVector((0 until dh.length).map(j => dhMasked(j) * cache.z(j)).toArray)

    val dzPre = dz *:* sigmoidGradVec(cache.z)
    val dnPre = dn *:* tanhGradVec(cache.n)

    // Gradient through reset gate
    val rh   = DenseVector(cache.r.toArray.zip(cache.hPrev.toArray).map { case (ri, hi) => ri * hi })
    val drh  = weights.wn.Wh.t * dnPre
    val dr   = DenseVector(drh.toArray.zip(cache.hPrev.toArray).map { case (di, hi) => di * hi })
    val drPre = dr *:* sigmoidGradVec(cache.r)
    val dhPrevFromR = DenseVector(drh.toArray.zip(cache.r.toArray).map { case (di, ri) => di * ri })

    val dhPrev = dhPrevFromH + dhPrevFromR +
                 (weights.wr.Wh.t * drPre) + (weights.wz.Wh.t * dzPre)
    val dx     = (weights.wr.Wx.t * drPre) + (weights.wz.Wx.t * dzPre) +
                 (weights.wn.Wx.t * dnPre)

    val grads = GruGrads(
      wr = GateGrads.compute(drPre, cache.x, cache.hPrev),
      wz = GateGrads.compute(dzPre, cache.x, cache.hPrev),
      wn = GateGrads.compute(dnPre, cache.x, rh)
    )
    (dx, dhPrev, grads)
  }
}
