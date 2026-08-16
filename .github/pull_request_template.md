## Summary

<!-- What does this change and why? Link any issue it closes. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance
- [ ] Documentation
- [ ] Build, CI, or tooling

## Numerical impact

<!--
Delete this section for pure docs or tooling changes.

If this touches the algorithm, distributions, or optimizer, say what happens to
the numbers. "None" is a fine answer, but say so explicitly, and say how you
checked: a comparison against the previous implementation, an analytical
result, or a test that pins the behaviour.
-->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy` passes
- [ ] `pytest` passes, and `pytest -m ""` if the change could affect slow tests
- [ ] New tests set `random_state` (or use the `rng` / `seed` fixtures) so they are deterministic
- [ ] `CHANGELOG.md` updated under `Unreleased`
