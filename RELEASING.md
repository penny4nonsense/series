# Releasing `series` to Maven Central

This document is the end-to-end checklist for publishing a release of
`series` to Maven Central via the Sonatype Central Portal. Steps 1–4 are
one-time setup; step 5 onward is repeated for each release.

The artifact coordinates are:

    io.github.penny4nonsense %% series % <version>

`series` cross-builds for Scala 2.13 and Scala 3. All publish commands use
the `+` prefix to release both.

---

## One-time setup

### 1. GitHub repository

The POM's `scmInfo` and `homepage` point at
`https://github.com/penny4nonsense/series`. That repository must exist and
be public before publishing — Maven Central validates the URLs resolve.

    git init
    git add .
    git commit -m "Initial public release of series"
    git branch -M main
    git remote add origin https://github.com/penny4nonsense/series.git
    git push -u origin main

Do NOT commit credentials, GPG private keys, or `~/.sbt` files.

### 2. Sonatype Central Portal account + namespace

1. Go to https://central.sonatype.com and sign in **with GitHub**.
2. Signing in via GitHub auto-verifies the namespace `io.github.penny4nonsense`.
   (No DNS TXT record needed — this is the advantage of the `io.github.*`
   namespace over a custom domain.)
3. Confirm the namespace shows as **verified** in the portal under
   "Namespaces".

### 3. Generate a publishing user token

1. In the Central Portal, go to your account → "Generate User Token".
2. This produces a token *username* and *password* (NOT your login
   password). Save both.

### 4. GPG key (for artifact signing)

Maven Central requires every artifact to be GPG-signed.

Install GnuPG (https://www.gnupg.org/download/), then:

    # Generate a key pair (choose RSA, 4096 bits, no expiry or a long expiry)
    gpg --full-generate-key

    # List keys to find your key ID (the long hex string)
    gpg --list-secret-keys --keyid-format=long

    # Publish the PUBLIC key to a keyserver so Sonatype can verify signatures
    gpg --keyserver keyserver.ubuntu.com --send-keys <YOUR_KEY_ID>

    # (optional, some validators check multiple servers)
    gpg --keyserver keys.openpgp.org --send-keys <YOUR_KEY_ID>

Remember the passphrase you set — sbt will prompt for it (or you can
configure it; see below).

---

## Per-release steps

### 5. Re-enable publishing config

These were disabled during development to keep plugin resolution from
interfering with the cross-build. Turn them back on:

**`project/plugins.sbt`** — add the signing plugin:

    addSbtPlugin("com.github.sbt" % "sbt-pgp" % "2.3.1")

**`build.sbt`** — uncomment the `publishTo` block:

    ThisBuild / publishTo := {
      val centralSnapshots = "https://central.sonatype.com/repository/maven-snapshots/"
      if (isSnapshot.value) Some("central-snapshots" at centralSnapshots)
      else localStaging.value
    }

### 6. Store credentials (outside the repo)

Create `~/.sbt/1.0/sonatype.sbt` (this path is global, never in the repo):

    credentials += Credentials(
      "Sonatype Nexus Repository Manager",
      "central.sonatype.com",
      "<TOKEN_USERNAME>",   // from step 3
      "<TOKEN_PASSWORD>"    // from step 3
    )

### 7. Set the release version

In `build.sbt`, set a non-SNAPSHOT version:

    ThisBuild / version := "0.1.0"

Follow semantic versioning (early-semver is already configured):
- 0.1.0 → first release
- 0.1.1 → backward-compatible bug fixes
- 0.2.0 → backward-compatible new features
- 1.0.0 → first stable API commitment

### 8. Final verification before publishing

    sbt "++2.13.18 test"
    sbt "++3.3.6 test"

Both must be green (currently 449/449 on each).

### 9. Build and sign artifacts for both Scala versions

    sbt +publishSigned

This creates signed artifacts (.jar, -sources.jar, -javadoc.jar, .pom,
and .asc signatures) in the local staging directory. sbt will prompt for
the GPG passphrase.

### 10. Release to Maven Central

With Central Portal publishing built into sbt 1.11+:

    sbt +publishSigned
    # then upload + release the staging bundle to the Central Portal

Check the Central Portal "Deployments" tab. The deployment will validate
(checks signatures, POM completeness, javadoc/sources presence). Once it
passes validation you can release it; it syncs to Maven Central within
~10–30 minutes, and appears in search within a few hours.

---

## Post-release

1. Tag the release in git:

       git tag -a v0.1.0 -m "series 0.1.0"
       git push origin v0.1.0

2. Bump to the next snapshot in `build.sbt`:

       ThisBuild / version := "0.2.0-SNAPSHOT"

3. (Optional) Disable the publish config again during development if you
   find the cross-build plugin resolution annoying, per the notes in
   build.sbt.

---

## Checklist for a clean first release

- [ ] GitHub repo exists and is public, code pushed
- [ ] Central Portal account created via GitHub
- [ ] Namespace `io.github.penny4nonsense` verified
- [ ] User token generated and saved
- [ ] GPG key generated, public key on keyserver
- [ ] `sbt-pgp` added to plugins.sbt
- [ ] `publishTo` uncommented in build.sbt
- [ ] Credentials in ~/.sbt/1.0/sonatype.sbt
- [ ] Version set to 0.1.0 (no -SNAPSHOT)
- [ ] `sbt "++2.13.18 test"` green
- [ ] `sbt "++3.3.6 test"` green
- [ ] `sbt +publishSigned` succeeds
- [ ] Deployment validates and released on Central Portal
- [ ] git tag pushed
