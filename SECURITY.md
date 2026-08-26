# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| latest (main) | ✅ |

Only the latest commit on `master` is supported. We don't backport fixes.

## Reporting a vulnerability

**Do not open a public issue for security bugs.**

Email [207921092+Hot-Coco@users.noreply.github.com](mailto:207921092+Hot-Coco@users.noreply.github.com) with details. You'll get a response within 48 hours.

## Scope

- Unsafe code in SIMD kernels — bounds violations, UB
- Model file parsing — buffer overflows, integer overflow
- Tokenizer — denial of service via malicious input

## Out of scope

- Model hallucinations, prompt injection, jailbreaking — these are ML problems, not Hypno bugs
- Performance DoS via extremely large models (that's what resource limits are for)
