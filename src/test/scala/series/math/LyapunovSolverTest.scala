package series.math

import breeze.linalg._
import org.scalatest.funsuite.AnyFunSuite

class LyapunovSolverTest extends AnyFunSuite {

  test("solution satisfies P = F * P * F.t + Q") {
    val F = DenseMatrix((0.5, 0.2), (0.0, 0.3))
    val Q = DenseMatrix.eye[Double](2) * 0.1

    val P = LyapunovSolver.solve(F, Q)
    val residual = P - (F * P * F.t + Q)
    assert(norm(residual.toDenseVector) < 1e-8)
  }

  test("solution is symmetric") {
    val F = DenseMatrix((0.5, 0.2), (0.0, 0.3))
    val Q = DenseMatrix.eye[Double](2) * 0.1

    val P = LyapunovSolver.solve(F, Q)
    val asymmetry = P - P.t
    assert(norm(asymmetry.toDenseVector) < 1e-10)
  }

  test("solution is positive definite") {
    val F = DenseMatrix((0.5, 0.2), (0.0, 0.3))
    val Q = DenseMatrix.eye[Double](2) * 0.1

    val P = LyapunovSolver.solve(F, Q)
    // All eigenvalues should be positive
    val eigs = eigSym(P)
    assert(eigs.eigenvalues.forall(_ > 0.0))
  }

  test("scalar case matches analytical solution") {
    // For scalar: P = f^2 * P + q => P = q / (1 - f^2)
    val f = 0.5
    val q = 1.0
    val F = DenseMatrix((f))
    val Q = DenseMatrix((q))

    val P = LyapunovSolver.solve(F, Q)
    val analytical = q / (1 - f * f)
    assert(math.abs(P(0, 0) - analytical) < 1e-8)
  }

  test("converges for near-unit-root process") {
    val F = DenseMatrix((0.99, 0.0), (0.0, 0.5))
    val Q = DenseMatrix.eye[Double](2) * 0.01

    val P = LyapunovSolver.solve(F, Q)
    val residual = P - (F * P * F.t + Q)
    assert(norm(residual.toDenseVector) < 1e-6)
  }
}