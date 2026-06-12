// ── Project coordinates ──────────────────────────────────────────────────────

ThisBuild / organization := "io.github.penny4nonsense"
ThisBuild / version      := "0.1.0"
ThisBuild / versionScheme := Some("early-semver")

// ── Scala versions ───────────────────────────────────────────────────────────

val scala213 = "2.13.18"
val scala3   = "3.3.6"   // latest LTS as of writing; adjust if needed

ThisBuild / scalaVersion       := scala213
ThisBuild / crossScalaVersions := Seq(scala213, scala3)

// ── POM / publishing metadata (required by Maven Central) ────────────────────

ThisBuild / homepage := Some(url("https://github.com/penny4nonsense/series"))
ThisBuild / licenses := List(
  "MIT" -> url("https://opensource.org/licenses/MIT")
)
ThisBuild / developers := List(
  Developer(
    id    = "penny4nonsense",
    name  = "Jason Parker",
    email = "",  // optional; fill in if you want it public
    url   = url("https://github.com/penny4nonsense")
  )
)
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/penny4nonsense/series"),
    "scm:git:https://github.com/penny4nonsense/series.git",
    "scm:git:git@github.com:penny4nonsense/series.git"
  )
)
ThisBuild / description :=
  "A from-scratch time series library for Scala: ARIMA/SARIMA, ETS, GARCH, " +
  "VAR/SVAR/VECM, Kalman filtering and smoothing, signal filters (HP, " +
  "Baxter-King, Christiano-Fitzgerald, Butterworth), decomposition (Classical, " +
  "STL, X-11), interpolation, and gradient-checked RNNs (Vanilla/LSTM/GRU). " +
  "Built on Breeze with no other dependencies."

ThisBuild / publishMavenStyle := true
ThisBuild / pomIncludeRepository := { _ => false }

// ── Publish target (uncomment at release time) ───────────────────────────────
// Central Portal publishing is built into sbt 1.11+. Uncomment this block
// when ready to publish (and re-add sbt-pgp to project/plugins.sbt).
//
ThisBuild / publishTo := {
  val centralSnapshots = "https://central.sonatype.com/repository/maven-snapshots/"
  if (isSnapshot.value) Some("central-snapshots" at centralSnapshots)
  else localStaging.value
}

// ── Build ────────────────────────────────────────────────────────────────────

lazy val root = (project in file("."))
  .settings(
    name := "series",
    libraryDependencies ++= Seq(
      "org.scalanlp"  %% "breeze"    % "2.1.0",
      "org.scalatest" %% "scalatest" % "3.2.18" % Test
    ),
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked"
    ),
    // Version-specific options
    scalacOptions ++= {
      CrossVersion.partialVersion(scalaVersion.value) match {
        case Some((3, _)) => Seq("-source:3.3")
        case _            => Seq("-Xsource:3")  // ease 2.13 -> 3 migration
      }
    }
  )
