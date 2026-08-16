# Security Policy

## Supported versions

Safe-ICE is pre-1.0 research software. Fixes are applied to the latest release
only; there are no maintained backport branches.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub's
[security advisory form](https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice/security/advisories/new),
or by email to <dfr@esmad.ipp.pt>.

Please include:

- a description of the issue and why you believe it is a security problem,
- the version or commit you tested,
- steps to reproduce, ideally a minimal script,
- any suggested fix, if you have one.

You can expect an acknowledgement within **7 days** and an assessment within
**30 days**. This is a small volunteer-maintained project, so please treat
those as good-faith targets rather than guarantees.

## Scope

Safe-ICE is a numerical library. It does not open network connections, listen
on sockets, or handle authentication, so the realistic attack surface is
narrow. Reports that are in scope include:

- code execution triggered by loading or deserialising untrusted input,
- a dependency vulnerability that Safe-ICE actually exposes to callers,
- anything that lets a user-supplied limit-state function escape the sandbox
  a caller reasonably expected.

The following are **not** security issues, though bug reports are welcome:

- inaccurate probability estimates or poor convergence,
- resource exhaustion caused by a caller's own choice of sample count or
  dimension,
- crashes from deliberately malformed arguments in a caller's own process.

## Disclosure

Please give us a chance to ship a fix before publishing details. We will credit
reporters in the release notes unless you ask us not to.
